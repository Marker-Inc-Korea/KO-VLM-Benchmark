import pytest
from vllm import LLM as vllm_LLM
from vllm import SamplingParams
from vllm_mock import LLM
from typing import Any

import asyncio
import time
from llama_index.core.multi_modal_llms import MultiModalLLM
from llama_index.core.schema import ImageNode
from llama_index.core.base.llms.types import CompletionResponse

from ko_vlm_benchmark.multi_page.verifier import verify_multipage_question

class MockMultiModalLLM(MultiModalLLM):
    """
    Custom Mock for MultiModalLLM since LlamaIndex's standard MockLLM 
    doesn't support image arguments in acomplete().
    """
    max_new_tokens: int = 10
    _mock_responses: list[str] = ["yes", "no", "yes"] 
    _call_count: int = 0

    def __init__(self, responses: list[str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        if responses:
            self._mock_responses = responses

    @property
    def metadata(self) -> Any:
        return {"model_name": "mock_vlm_pytest"}

    def complete(self, prompt: str, image_documents: list[ImageNode], **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text="yes")

    async def acomplete(self, prompt: str, image_documents: list[ImageNode], **kwargs: Any) -> CompletionResponse:
        # Simulate network latency to verify async concurrency
        await asyncio.sleep(0.1)
        
        # Cycle responses to deterministicly test majority voting
        response_text = self._mock_responses[self._call_count % len(self._mock_responses)]
        self._call_count += 1
        return CompletionResponse(text=response_text)

    def stream_complete(self, *args, **kwargs): raise NotImplementedError
    async def astream_complete(self, *args, **kwargs): raise NotImplementedError
    def chat(self, *args, **kwargs): raise NotImplementedError
    async def achat(self, *args, **kwargs): raise NotImplementedError
    def stream_chat(self, *args, **kwargs): raise NotImplementedError
    async def astream_chat(self, *args, **kwargs): raise NotImplementedError

# -----------------------------------------------------------------------------
# Test Logic
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verify_multipage_question_concurrency_and_voting():
    """
    Tests verify_multipage_question for:
    1. Correct Majority Voting (2 vs 1).
    2. Async Concurrency (execution time < serial time).
    """
    # Setup
    VOTES_PER_QUERY = 3
    TOTAL_QUERIES = 20
    MAX_CONCURRENT = 5
    
    # Mock returns ["yes", "no", "yes"] -> Majority should always be "yes"
    mock_llm = MockMultiModalLLM(responses=["yes", "no", "yes"])
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Dummy Data
    queries = [f"Test Query {i}" for i in range(TOTAL_QUERIES)]
    images = [ImageNode(image_path="dummy.jpg")]

    # Execution
    start_time = time.perf_counter()

    tasks = [
        verify_multipage_question(
            mm_llm=mock_llm,
            query=q,
            image_documents=images,
            semaphore=semaphore,
            vote_count=VOTES_PER_QUERY,
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
    expected_calls = TOTAL_QUERIES * VOTES_PER_QUERY
    assert mock_llm._call_count == expected_calls, f"Expected {expected_calls} calls, got {mock_llm._call_count}"

    # 3. Verify Concurrency
    # Serial time would be (60 calls * 0.1s) = 6.0s
    # We expect parallel execution to be significantly faster
    serial_time_estimate = expected_calls * 0.1
    assert duration < (serial_time_estimate / 2), \
        f"Concurrency not detected! Duration {duration:.2f}s is too close to serial {serial_time_estimate}s"



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
