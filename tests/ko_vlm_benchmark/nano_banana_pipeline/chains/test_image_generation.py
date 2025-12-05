"""Minimal test for image_generation chain with real Gemini client."""

import os
from pathlib import Path

import pytest

from ko_vlm_benchmark.nano_banana_pipeline.chains.image_generation import (
    ImageGenerationChain,
)
from ko_vlm_benchmark.nano_banana_pipeline.config import PipelineConfig


@pytest.fixture
def config(tmp_path: Path) -> PipelineConfig:
    """Create pipeline config with API key from environment."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    return PipelineConfig(
        google_api_key=api_key,
        gemini_image_model="gemini-2.5-flash-image",
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def chain(config: PipelineConfig) -> ImageGenerationChain:
    """Create image generation chain."""
    return ImageGenerationChain(config)


@pytest.fixture
def test_image(tmp_path: Path) -> Path:
    """Create a simple test image."""
    from PIL import Image

    image_path = tmp_path / "test_document.png"
    img = Image.new("RGB", (200, 100), color="white")
    img.save(image_path)
    return image_path


def test_image_generation_invoke(chain: ImageGenerationChain, test_image: Path) -> None:
    """Test that image generation chain runs and returns valid output."""
    image_prompt = "A simple document with the title 'Test Report' and a blue header."

    result = chain.invoke(test_image, image_prompt)

    assert "image_bytes" in result
    assert "image_path" in result
    assert isinstance(result["image_bytes"], bytes)
    assert len(result["image_bytes"]) > 0
    assert Path(result["image_path"]).exists()
