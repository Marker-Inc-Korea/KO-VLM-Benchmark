import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest
from vllm import LLM as vllm_LLM
from vllm import SamplingParams
from vllm_mock import LLM

from ko_vlm_benchmark.multi_page.verifier import verify_multipage_question


def create_mock_claude_acomplete(responses: list[str]):
    """
    Factory function to create a mock for claude_multimodal_acomplete
    that cycles through predetermined responses and simulates network latency.
    """
    call_count = 0

    async def mock_acomplete(api_key: str, image_path_list: list[str], user_text: str) -> str:
        nonlocal call_count
        # Simulate network latency to verify async concurrency
        await asyncio.sleep(0.1)

        # Cycle responses to deterministically test majority voting
        response_text = responses[call_count % len(responses)]
        call_count += 1
        return response_text

    # Wrap in AsyncMock to enable tracking
    mock = AsyncMock(side_effect=mock_acomplete)
    return mock


# -----------------------------------------------------------------------------
# Test Logic
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
@patch("ko_vlm_benchmark.multi_page.verifier.claude_multimodal_acomplete")
async def test_verify_multipage_question_concurrency_and_voting(mock_claude_acomplete):
    """
    Tests verify_multipage_question for:
    1. Correct Majority Voting (2 vs 1).
    2. Async Concurrency (execution time < serial time).
    """
    # Setup
    votes_per_query = 3
    total_queries = 20
    max_concurrent = 5

    # Mock returns ["yes", "no", "yes"] -> Majority should always be "yes"
    mock_claude_acomplete.side_effect = create_mock_claude_acomplete(responses=["yes", "no", "yes"]).side_effect
    semaphore = asyncio.Semaphore(max_concurrent)

    # Dummy Data
    queries = [f"Test Query {i}" for i in range(total_queries)]
    images = ["dummy.jpg"]

    # Execution
    start_time = time.perf_counter()

    tasks = [
        verify_multipage_question(
            api_key="test_api_key",
            query=q,
            image_paths=images,
            semaphore=semaphore,
            vote_count=votes_per_query,
        )
        for q in queries
    ]

    decisions = await asyncio.gather(*tasks)

    duration = time.perf_counter() - start_time

    # Assertions

    # 1. Verify Voting Logic
    # With ["yes", "no", "yes"], every result must be "yes"
    assert all(d == "yes" for d in decisions), "Majority voting failed: expected all 'yes'"

    # 2. Verify API Call Count
    expected_calls = total_queries * votes_per_query
    assert mock_claude_acomplete.call_count == expected_calls, (
        f"Expected {expected_calls} calls, got {mock_claude_acomplete.call_count}"
    )

    # 3. Verify Concurrency
    # Serial time would be (60 calls * 0.1s) = 6.0s
    # We expect parallel execution to be significantly faster
    serial_time_estimate = expected_calls * 0.1
    assert duration < (serial_time_estimate / 2), (
        f"Concurrency not detected! Duration {duration:.2f}s is too close to serial {serial_time_estimate}s"
    )


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
