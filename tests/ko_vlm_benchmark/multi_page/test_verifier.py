import pytest
from vllm import LLM as vllm_LLM
from vllm import SamplingParams
from vllm_mock import LLM


def test_calculate_prompt_logprobs():
    from ko_vlm_benchmark.multi_page.verifier import calculate_prompt_logprobs

    llm = LLM("mock-model")
    result = llm.generate(["This is a test prompt.", "Test prompt 2"], SamplingParams(prompt_logprobs=1))
    assert len(result) == 2

    for response in result:
        prompt_logprobs = response.prompt_logprobs
        target_token_indices = [[2, 3]]  # 계산하고 싶은 토큰 인덱스 그룹 설정.
        prompt_logprob_res = calculate_prompt_logprobs(prompt_logprobs, target_token_indices)
        assert isinstance(prompt_logprob_res, float)
        assert -10.0 <= prompt_logprob_res <= 0.0


@pytest.mark.gpu
def test_verify_two_hop_queries():
    queries = [
        "What is the capital of France and who is its current president?",
        "Name the largest planet in our solar system and its most famous moon.",
    ]

    documents = [
        ("France's capital is Paris.", "The current president of France is Emmanuel Macron."),
        ("The largest planet in our solar system is Jupiter.", "Jupiter's most famous moon is Ganymede."),
    ]

    answers = [
        "The capital of France is Paris and its current president is Emmanuel Macron.",
        "The largest planet in our solar system is Jupiter and its most famous moon is Ganymede.",
    ]

    llm = vllm_LLM("facebook/opt-125m")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    from ko_vlm_benchmark.multi_page.verifier import verify_two_hop_queries

    balance_scores = verify_two_hop_queries(queries, documents, answers, llm, tokenizer)
    assert len(balance_scores) == len(queries)
    assert all(isinstance(x, float) for x in balance_scores)
