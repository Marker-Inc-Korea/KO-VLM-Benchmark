from vllm import SamplingParams
from vllm_mock import LLM


def test_calculate_prompt_logprobs(monkeypatch):
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
