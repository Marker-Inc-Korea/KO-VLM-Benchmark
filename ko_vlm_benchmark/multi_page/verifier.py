from vllm.logprobs import Logprob


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
