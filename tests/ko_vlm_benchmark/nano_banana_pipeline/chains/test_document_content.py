"""Minimal test for document_content chain with real Anthropic client."""

import os

import pytest

from ko_vlm_benchmark.nano_banana_pipeline.chains.document_content import (
    DocumentContentChain,
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
def chain(config: PipelineConfig) -> DocumentContentChain:
    """Create document content chain."""
    return DocumentContentChain(config)


def test_document_content_invoke(chain: DocumentContentChain) -> None:
    """Test that document content chain returns valid structured output."""
    multi_hop_question = "걸그룹 에스파와 뉴진스의 멤버 수를 더하면 총 몇 명 입니까?"
    multi_hop_answer = "에스파는 4명, 뉴진스는 5명이므로 총 9명입니다."
    additional_info = (
        "뉴진스의 멤버들에 대한 정보가 필요하다. 뉴진스는 민지, 하니, 다니엘, 해린, 혜인으로 구성되어 있다."
    )
    visual_description = "이것은 한국의 K-Pop 걸그룹 에스파에 대한 정보를 담고 있는 위키백과 문서이다. 에스파의 데뷔 연도, 앨범, 프로필 사진, 멤버 정보 들이 자세히 묘사되어 있는 웹페이지이다."

    result = chain.invoke(multi_hop_question, multi_hop_answer, additional_info, visual_description)

    assert "document_content" in result
    assert isinstance(result["document_content"], str)
    assert len(result["document_content"]) > 0
