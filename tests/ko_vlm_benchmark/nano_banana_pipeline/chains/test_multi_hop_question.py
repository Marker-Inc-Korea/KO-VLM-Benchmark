"""Minimal test for multi_hop_question chain with real Anthropic client."""

import os

import pytest

from ko_vlm_benchmark.nano_banana_pipeline.chains.multi_hop_question import (
    MultiHopQuestionChain,
)
from ko_vlm_benchmark.nano_banana_pipeline.config import PipelineConfig
from ko_vlm_benchmark.nano_banana_pipeline.types import SingleHopResult


@pytest.fixture
def config() -> PipelineConfig:
    """Create pipeline config with API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return PipelineConfig(anthropic_api_key=api_key)


@pytest.fixture
def chain(config: PipelineConfig) -> MultiHopQuestionChain:
    """Create multi-hop question chain."""
    return MultiHopQuestionChain(config)


def test_multi_hop_question_invoke(chain: MultiHopQuestionChain) -> None:
    """Test that multi-hop question chain returns valid structured output."""
    visual_description = "This is a Q3 2024 sales report. Total revenue is 15 billion won, up 15% year-over-year."
    single_hop_result = SingleHopResult(
        question="What is the total revenue for Q3 2024?",
        answer="15 billion won.",
        reasoning="The document states that total revenue is 15 billion won.",
    )

    result = chain.invoke(single_hop_result, visual_description)

    assert "multi_hop_question" in result
    assert "additional_info_needed" in result
    assert isinstance(result["multi_hop_question"], str)
    assert isinstance(result["additional_info_needed"], str)
    assert len(result["multi_hop_question"]) > 0
