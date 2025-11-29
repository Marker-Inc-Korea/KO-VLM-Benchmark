import os
from typing import Any

import click
import fire
import pandas as pd
from llama_index.core.base.llms.base import BaseLLM
from llama_index.llms.anthropic import Anthropic
from vllm import LLM

from ko_vlm_benchmark.multi_choice.generate_gemini import (
    generate_wrong_answer_explain,
    generate_wrong_answer_meaning,
    generate_wrong_answer_numerical,
)

"""
from ko_vlm_benchmark.multi_choice.generate_anthropic import (
    generate_wrong_answer_numerical,
    generate_wrong_answer_meaning,
    generate_wrong_answer_explain
)
"""


def initialize_models(
    # vllm_model_name: str,
    llm_model_name: str,
) -> tuple[LLM, Any, BaseLLM]:
    """Initialize all models at once to avoid redundant overhead."""
    # Initialize vLLM for logprobs verification
    # click.echo(f"Initializing vLLM model: {vllm_model_name}")
    # vllm_llm = LLM(model=vllm_model_name, tensor_parallel_size=1, gpu_memory_utilization=0.8)
    # tokenizer = vllm_llm.get_tokenizer()

    # Initialize LlamaIndex BaseLLM for question generation
    click.echo(f"Initializing BaseLLM: {llm_model_name}")
    base_llm = Anthropic(model=llm_model_name)

    return base_llm


# generate multi choice question
# there is desired answer
def main(
    dataset_path="./data_multi_page/results_sub_1.xlsx",
    llm_model="claude-sonnet-4-5-20250929",
    vllm_model="Qwen/Qwen3-8B",
    api_key=None,
    save_path="./data_multi_page/results_multi_choice_sub_1.xlsx",
):

    # semaphore = asyncio.Semaphore(16)

    # load model (if load claude, use this)
    # base_llm = initialize_models(
    #    # vllm_model_name=vllm_model,
    #    llm_model_name=llm_model,
    # )

    # load pandas
    df = pd.read_excel(dataset_path)

    # new df
    check_df = os.listdir("./data_multi_page")
    if "results_multi_choice_sub_1.xlsx" in check_df:
        new_df = pd.read_excel(save_path)
        row_count = len(new_df)
    else:
        new_df = pd.DataFrame(
            columns=[*df.columns.tolist(), "model_wrong_answer1", "model_wrong_answer2", "model_wrong_answer3"]
        )
        row_count = 0
    print("columns:", new_df.columns)

    # (kyujin) 11/29 gemini version
    # make 4 wrong answers
    for i in range(len(df)):
        if (i + 1) <= row_count:
            continue

        model_answer_status = df.iloc[i].model_answer_status.strip()
        if model_answer_status == "success":
            # 기본정보 불러오기
            image_path1 = df.iloc[i].page_1_image.strip()
            image_path2 = df.iloc[i].page_2_image.strip()

            image1_context = df.iloc[i].page_1_context.strip()
            image2_context = df.iloc[i].page_2_context.strip()

            question = df.iloc[i].generated_question.strip()
            gt_answer = df.iloc[i].model_answer.strip()

            ######## we want hard cases (currently, not using context information)
            # first, make wrong answer (수치적)
            wrong_answer1 = generate_wrong_answer_numerical(
                image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
            )
            print("## wrong_answer1:\n", wrong_answer1)

            # second, make wrong answer (의미적)
            wrong_answer2 = generate_wrong_answer_meaning(
                image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
            )
            print("## wrong_answer2:\n", wrong_answer2)

            # third, maek wrong answer (풀이과정)
            wrong_answer3 = generate_wrong_answer_explain(
                image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
            )
            print("## wrong_answer3:\n", wrong_answer3)

            ######### Verify, is it hard case?

        else:
            # print('skip this row. because of failed case')
            wrong_answer1 = ""
            wrong_answer2 = ""
            wrong_answer3 = ""
            pass

        # saving
        new_row = df.iloc[i].tolist()
        new_row = [*new_row, wrong_answer1, wrong_answer2, wrong_answer3]

        new_df.loc[i] = new_row
        new_df.to_excel(save_path, index=False)

        break

    print("## finish")


if __name__ == "__main__":
    fire.Fire(main)
