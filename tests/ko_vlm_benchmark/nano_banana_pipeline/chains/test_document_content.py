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
    multi_hop_question = "How does the Q3 2024 revenue of 15 billion won compare to the industry average?"
    additional_info_needed = "Industry average revenue for Q3 2024 in the same sector."

    result = chain.invoke(multi_hop_question, additional_info_needed)

    assert "document_content" in result
    assert "search_results" in result
    assert isinstance(result["document_content"], str)
    assert isinstance(result["search_results"], list)
    assert len(result["document_content"]) > 0
