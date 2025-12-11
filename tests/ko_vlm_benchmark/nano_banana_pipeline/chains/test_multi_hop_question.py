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
    visual_description = "이것은 한국의 K-Pop 걸그룹 에스파에 대한 정보를 담고 있는 위키백과 문서이다. 에스파의 데뷔 연도, 앨범, 프로필 사진, 멤버 정보 들이 자세히 묘사되어 있는 웹페이지이다."
    single_hop_result = SingleHopResult(
        question="에스파의 멤버 수는 몇 명입니까?",
        answer="에스파는 총 4명으로, 카리나, 지젤, 윈터, 닝닝으로 구성되어 있습니다.",
        reasoning="문서에 에스파의 멤버가 카리나, 지젤, 윈터, 닝닝으로 명시되어 있으며 이는 총 4명입니다.",
    )

    result = chain.invoke(single_hop_result, visual_description)

    assert "multi_hop_question" in result
    assert "multi_hop_answer" in result
    assert "additional_info" in result
    assert "question_style" in result
    assert "search_queries" in result
    assert "search_results" in result
    assert isinstance(result["multi_hop_question"], str)
    assert isinstance(result["multi_hop_answer"], str)
    assert isinstance(result["additional_info"], str)
    assert isinstance(result["question_style"], str)
    assert isinstance(result["search_queries"], list)
    assert isinstance(result["search_results"], list)
    assert len(result["multi_hop_question"]) > 0
    assert len(result["multi_hop_answer"]) > 0
    assert len(result["additional_info"]) > 0
    assert len(result["question_style"]) > 0
    assert len(result["search_queries"]) > 0, "Web search should have been performed"
    for query in result["search_queries"]:
        assert isinstance(query, str)
        assert len(query) > 0
