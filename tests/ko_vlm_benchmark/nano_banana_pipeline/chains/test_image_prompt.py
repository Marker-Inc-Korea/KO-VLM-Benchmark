"""Minimal test for image_prompt chain with real Anthropic client."""

import os

import pytest

from ko_vlm_benchmark.nano_banana_pipeline.chains.image_prompt import (
    ImagePromptChain,
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
def chain(config: PipelineConfig) -> ImagePromptChain:
    """Create image prompt chain."""
    return ImagePromptChain(config)


def test_image_prompt_invoke(chain: ImagePromptChain) -> None:
    """Test that image prompt chain returns valid structured output."""
    visual_description = "A formal business report with a blue header, company logo in the top left corner, and a bar chart showing quarterly sales data."
    document_content = "Q3 2024 Sales Report\n\nTotal Revenue: 15 billion won\nYear-over-Year Growth: 15%\n\nKey Highlights:\n- Strong performance in electronics division\n- Expansion into new markets"

    result = chain.invoke(visual_description, document_content)

    assert "image_prompt" in result
    assert "style_description" in result
    assert isinstance(result["image_prompt"], str)
    assert isinstance(result["style_description"], str)
    assert len(result["image_prompt"]) > 0
    assert len(result["style_description"]) > 0
