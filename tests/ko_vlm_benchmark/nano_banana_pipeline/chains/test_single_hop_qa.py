"""Minimal test for single_hop_qa chain with real Anthropic client."""

import os

import pytest

from ko_vlm_benchmark.nano_banana_pipeline.chains.single_hop_qa import (
    SingleHopQAChain,
)
from ko_vlm_benchmark.nano_banana_pipeline.config import PipelineConfig


@pytest.fixture
def config() -> PipelineConfig:
    """Create pipeline config with API key from environment."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return PipelineConfig(anthropic_api_key=api_key)


@pytest.fixture
def chain(config: PipelineConfig) -> SingleHopQAChain:
    """Create single-hop QA chain."""
    return SingleHopQAChain(config)


def test_single_hop_qa_invoke(chain: SingleHopQAChain) -> None:
    """Test that single-hop QA chain returns valid structured output."""
    visual_description = (
        "이 문서는 2024년 3분기 매출 보고서입니다. 총 매출액은 150억원이며, 전년 동기 대비 15% 증가했습니다."
    )

    result = chain.invoke(visual_description)

    assert "question" in result
    assert "answer" in result
    assert "reasoning" in result
    assert isinstance(result["question"], str)
    assert isinstance(result["answer"], str)
    assert isinstance(result["reasoning"], str)
    assert len(result["question"]) > 0
    assert len(result["answer"]) > 0
