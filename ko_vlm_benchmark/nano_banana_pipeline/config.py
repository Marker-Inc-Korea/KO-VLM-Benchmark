"""Configuration for the nano banana pipeline."""

import os
from pathlib import Path

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Configuration for the nano banana pipeline."""

    # API Keys (from environment by default)
    anthropic_api_key: str = Field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    google_api_key: str = Field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))

    # Model settings
    sonnet_model: str = "claude-sonnet-4-5-20250929"
    gemini_image_model: str = "gemini-2.5-flash-image"

    # Generation settings
    max_tokens: int = 16384
    temperature: float = 1.0

    # Web search settings
    web_search_max_uses: int = 5
    web_search_allowed_domains: list[str] | None = None

    # Output settings
    output_dir: Path = Field(default_factory=lambda: Path("output/generated_images"))

    # Batch mode settings
    skip_image_generation: bool = False  # Skip Step 5 for batch processing

    model_config = {"arbitrary_types_allowed": True}

    def ensure_output_dir(self) -> Path:
        """Ensure output directory exists and return it."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
