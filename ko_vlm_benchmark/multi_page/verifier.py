import asyncio
import re
from collections import Counter
from typing import Any, Literal

import pandas as pd
from nltk import PunktSentenceTokenizer
from vllm import LLM, SamplingParams
from vllm.logprobs import Logprob

from ko_vlm_benchmark.anthropic import claude_multimodal_acomplete


async def verify_multipage_question(
    api_key: str,
    query: str,
    image_paths: list[str],
    semaphore: asyncio.Semaphore,
    vote_count: int = 3,
) -> str:
    """
    Determines if a question requires multiple pages to answer using a VLM.
    Returns 'yes' or 'no' based on a majority vote of 3 attempts.
    """

    # Prompt designed to force a binary yes/no decision
    verification_prompt = (
        "You are a dataset verifier. Your task is to determine if the user's "
        "query requires information from MULTIPLE pages (images) to be answered correctly, "
        "or if it can be answered using only one of them.\n\n"
        f"Query: {query}\n\n"
        "Review the provided document images. Does answering this query require "
        "synthesizing information from more than one image? "
        "Return exactly and only the word 'yes' or 'no'."
    )

    async def single_request() -> str:
        """Helper to run a single async VLM request with semaphore protection."""
        async with semaphore:
            try:
                # acomplete is the async version of complete for LlamaIndex LLMs
                response = await claude_multimodal_acomplete(api_key, image_paths, verification_prompt)
                # Clean output to ensure we just get 'yes' or 'no'
                decision = response.strip().lower()
                if "yes" in decision:
                    return "yes"
                if "no" in decision:
                    return "no"
                else:
                    return "no"
            except Exception as e:
                print(f"Request failed: {e}")
                return "error"

    # Create tasks for self-consistency (VOTE_COUNT times)
    tasks = [single_request() for _ in range(vote_count)]

    # Run requests concurrently
    results = await asyncio.gather(*tasks)

    # Filter out errors
    valid_results = [r for r in results if r in ["yes", "no"]]

    if not valid_results:
        return "no"  # Default if all failed

    # Majority Vote
    vote_counts = Counter(valid_results)
    majority_decision, _ = vote_counts.most_common(1)[0]

    return majority_decision


def build_prompt(
    query: str,
    documents: tuple[str, str],
    answer: str,
    which_document: Literal["first", "second", "both"],
) -> list[dict]:
    if which_document == "first":
        docs_text = f"Document 1: {documents[0]}"
    elif which_document == "second":
        docs_text = f"Document 1: {documents[1]}"
    else:  # both
        docs_text = f"Document 1: {documents[0]}\nDocument 2: {documents[1]}"

    chat_prompt = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers the user question based on provided documents.",
        },
        {"role": "user", "content": f"{docs_text}\n\nQuestion: {query}"},
        {"role": "assistant", "content": answer},
    ]
    return chat_prompt


def verify_two_hop_queries(
    queries: list[str],
    documents: list[tuple[str, str]],
    answers: list[str],
    llm: LLM,
    tokenizer: Any,
    **sampling_kwargs: Any,
) -> list[float]:
    # Build a result dataframe
    df = pd.DataFrame({
        "query": queries,
        "document_1": [doc[0] for doc in documents],
        "document_2": [doc[1] for doc in documents],
        "answer": answers,
    })

    # Calculate answer lengths
    df["both_prompt"] = df.apply(
        lambda row: build_prompt(row["query"], (row["document_1"], row["document_2"]), row["answer"], "both"),
        axis=1,
    )
    df["first_prompt"] = df.apply(
        lambda row: build_prompt(row["query"], (row["document_1"], row["document_2"]), row["answer"], "first"),
        axis=1,
    )
    df["second_prompt"] = df.apply(
        lambda row: build_prompt(row["query"], (row["document_1"], row["document_2"]), row["answer"], "second"),
        axis=1,
    )

    df["both_logprob"] = calculate_logprobs(df["both_prompt"].tolist(), tokenizer=tokenizer, llm=llm, **sampling_kwargs)
    df["first_logprob"] = calculate_logprobs(
        df["first_prompt"].tolist(), tokenizer=tokenizer, llm=llm, **sampling_kwargs
    )
    df["second_logprob"] = calculate_logprobs(
        df["second_prompt"].tolist(), tokenizer=tokenizer, llm=llm, **sampling_kwargs
    )

    df["info_gain"] = df.apply(
        lambda row: calculate_info_gain(row["both_logprob"], row["first_logprob"], row["second_logprob"]),
        axis=1,
    )
    return df["info_gain"].tolist()


def calculate_info_gain(both_logprob: float, first_logprob: float, second_logprob: float) -> float:
    """
    Calculate information gain.
    The higher it is, the better (which means it is likely to be the multi-hop question)

    Uses simple difference between both_logprob and the max of individual logprobs.
    Since logprobs are negative (closer to 0 = higher probability), a positive
    difference indicates that having both documents improves answer quality.
    """
    # both_logprob should be higher (less negative) than individual documents
    # for a good multi-page question
    return both_logprob - max(first_logprob, second_logprob)


def calculate_prompt_logprobs(
    prompt_logprobs: list[dict[int, Logprob] | None], target_token_indices: list[list[int]]
) -> float:
    logprob_list = []
    for token_indices in target_token_indices:
        temp_logprobs = []
        target_logprobs_dict: list[dict[int, Logprob] | None] = [
            prompt_logprobs[i] for i in token_indices if prompt_logprobs[i] is not None
        ]
        for elem in target_logprobs_dict:
            if elem is not None:
                temp_logprobs.append(next(iter(elem.values())).logprob)
        if temp_logprobs:
            logprob_list.append(sum(temp_logprobs) / len(temp_logprobs))
    if not logprob_list:
        return -10.0
    return sum(logprob_list) / len(logprob_list)


def split_response(tokenizer: Any, conversation: list[dict]) -> tuple[str, list[dict]]:
    response = conversation[-1]["content"]
    sentences = text_split_by_punctuation(response, return_dict=False)

    formatted_prompt = tokenizer.apply_chat_template(conversation, tokenize=False, enable_thinking=False)

    targets = []
    for sentence in sentences:
        start_char = formatted_prompt.find(sentence)
        end_char = start_char + len(sentence)
        targets.append({"start_char": start_char, "end_char": end_char, "sentence": sentence})

    tokenized = tokenizer(formatted_prompt, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")

    offsets = tokenized.pop("offset_mapping")

    result_indices = []
    for target in targets:
        indices = [
            idx
            for idx, (start, end) in enumerate(offsets[0])
            if start >= target["start_char"] and end <= target["end_char"]
        ]
        result_indices.append(indices)

    return formatted_prompt, [
        {"token_indices": indices, "sentence": target["sentence"]}
        for indices, target in zip(result_indices, targets, strict=True)
    ]


def text_split_by_punctuation(original_text: str, return_dict: bool = False) -> list[str] | list[dict]:
    """
    Code from https://github.com/facebookresearch/SelfCite
    """
    # text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', original_text)  # separate period without space
    text = original_text
    custom_sent_tokenizer = PunktSentenceTokenizer()
    punctuations = r"([。；！？])"  # For Chinese support # noqa: RUF001

    separated = custom_sent_tokenizer.tokenize(text)
    separated = [item for s in separated for item in re.split(punctuations, s)]
    # Put the punctuations back to the sentence
    for i in range(1, len(separated)):
        if re.match(punctuations, separated[i]):
            separated[i - 1] += separated[i]
            separated[i] = ""

    separated = [s for s in separated if s != ""]
    if len(separated) == 1:
        separated = original_text.split("\n\n")
    separated = [s.strip() for s in separated if s.strip() != ""]
    if not return_dict:
        return separated
    else:
        pos = 0
        res = []
        for i, sent in enumerate(separated):
            st = original_text.find(sent, pos)
            ed = st + len(sent)
            res.append({
                "c_idx": i,
                "content": sent,
                "start_idx": st,
                "end_idx": ed,
            })
            pos = ed
        return res


def calculate_logprobs(input_conversation_list: list[dict], tokenizer: Any, llm: LLM, **sampling_kwargs) -> list[float]:
    df = pd.DataFrame({"conversation": input_conversation_list})
    df[["formatted_prompt", "target_dict"]] = df.apply(
        lambda row: split_response(tokenizer, row["conversation"]),
        result_type="expand",
        axis=1,
    )
    df["target_token_indices"] = df["target_dict"].apply(lambda x: [y["token_indices"] for y in x])
    df["target_sentences"] = df["target_dict"].apply(lambda x: [y["sentence"] for y in x])
    df.drop(columns=["target_dict"], inplace=True)

    outputs = llm.generate(
        df["formatted_prompt"].tolist(), sampling_params=SamplingParams(prompt_logprobs=1, **sampling_kwargs)
    )
    df["prompt_logprobs"] = [output.prompt_logprobs for output in outputs]

    df["logprob_score"] = df.apply(
        lambda row: calculate_prompt_logprobs(row["prompt_logprobs"], row["target_token_indices"]),
        axis=1,
    )
    return df["logprob_score"].tolist()
