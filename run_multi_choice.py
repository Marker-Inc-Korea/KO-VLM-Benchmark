import asyncio
import os
from typing import Literal

import fire
import pandas as pd

from ko_vlm_benchmark.multi_choice import generate_anthropic, generate_gemini


def generate_wrong_answers(
    model_type: Literal["gemini", "anthropic"],
    image_path1: str,
    image_path2: str,
    image1_context: str,
    image2_context: str,
    question: str,
    gt_answer: str,
    api_key: str,
) -> tuple[str, str, str]:
    """Generate three types of wrong answers using the specified model backend."""
    if model_type == "gemini":
        wrong_answer1 = generate_gemini.generate_wrong_answer_numerical(
            image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
        )
        wrong_answer2 = generate_gemini.generate_wrong_answer_meaning(
            image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
        )
        wrong_answer3 = generate_gemini.generate_wrong_answer_explain(
            image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
        )
    elif model_type == "anthropic":
        # Anthropic functions are async, so we need to run them in an event loop
        wrong_answer1 = asyncio.run(
            generate_anthropic.generate_wrong_answer_numerical(
                image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
            )
        )
        wrong_answer2 = asyncio.run(
            generate_anthropic.generate_wrong_answer_meaning(
                image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
            )
        )
        wrong_answer3 = asyncio.run(
            generate_anthropic.generate_wrong_answer_explain(
                image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
            )
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'gemini' or 'anthropic'.")

    return wrong_answer1, wrong_answer2, wrong_answer3


def main(
    dataset_path: str = "./data_multi_page/results_sub_1.xlsx",
    model_type: Literal["gemini", "anthropic"] = "gemini",
    api_key: str | None = None,
    save_path: str = "./data_multi_page/results_multi_choice_sub_1.xlsx",
):

    print(f"Using model type: {model_type}")

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

            # Generate wrong answers using the selected model backend
            wrong_answer1, wrong_answer2, wrong_answer3 = generate_wrong_answers(
                model_type=model_type,
                image_path1=image_path1,
                image_path2=image_path2,
                image1_context=image1_context,
                image2_context=image2_context,
                question=question,
                gt_answer=gt_answer,
                api_key=api_key,
            )
            print("## wrong_answer1 (numerical):\n", wrong_answer1)
            print("## wrong_answer2 (meaning):\n", wrong_answer2)
            print("## wrong_answer3 (explain):\n", wrong_answer3)

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
