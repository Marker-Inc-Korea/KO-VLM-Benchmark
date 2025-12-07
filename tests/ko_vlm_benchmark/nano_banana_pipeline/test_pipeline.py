"""Minimal test for NanoBananaPipeline with real API clients."""

import os
from pathlib import Path

import pytest

from ko_vlm_benchmark.nano_banana_pipeline.config import PipelineConfig
from ko_vlm_benchmark.nano_banana_pipeline.pipeline import NanoBananaPipeline
from ko_vlm_benchmark.nano_banana_pipeline.types import PipelineInput


@pytest.fixture
def config() -> PipelineConfig:
    """Create pipeline config with API keys from environment."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    if not anthropic_key or not google_key:
        pytest.skip("ANTHROPIC_API_KEY or GOOGLE_API_KEY not set")
    return PipelineConfig(
        anthropic_api_key=anthropic_key,
        google_api_key=google_key,
        skip_image_generation=True,
    )


@pytest.fixture
def pipeline(config: PipelineConfig) -> NanoBananaPipeline:
    """Create pipeline instance."""
    return NanoBananaPipeline(config)


@pytest.fixture
def test_image(tmp_path: Path) -> Path:
    """Create a simple test image."""
    from PIL import Image

    image_path = tmp_path / "test_doc.png"
    img = Image.new("RGB", (200, 100), color="white")
    img.save(image_path)
    return image_path


def test_pipeline_run(pipeline: NanoBananaPipeline, test_image: Path) -> None:
    """Test that pipeline runs and returns partial output."""
    input_data = PipelineInput(
        image_path=str(test_image),
        visual_description="삼성전자의 2024 3분기의 매출은 총 150억원이다.",
    )

    result = pipeline.run(input_data)

    assert "single_hop_question" in result
    assert "multi_hop_question" in result
    assert "multi_hop_question_style" in result
    assert "image_generation_prompt" in result
