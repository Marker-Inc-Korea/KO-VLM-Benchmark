"""Nano Banana Pipeline - Multi-hop Multi-page VQA Generation.

This package provides a LangChain/LCEL-based pipeline for generating
multi-hop, multi-page VQA benchmark data using:
- Claude Sonnet 4.5 with Anthropic's web search tool
- Gemini (Nano Banana Pro) for document image generation

Example usage:
    >>> from ko_vlm_benchmark.nano_banana_pipeline import (
    ...     NanoBananaPipeline,
    ...     PipelineConfig,
    ...     run_nano_banana_pipeline,
    ... )
    >>>
    >>> # Simple usage
    >>> result = await run_nano_banana_pipeline(
    ...     image_path="document.png",
    ...     visual_description="A financial report showing...",
    ...     doc_type="report",
    ... )
    >>>
    >>> # With custom config
    >>> config = PipelineConfig(
    ...     anthropic_api_key="your-key",
    ...     google_api_key="your-key",
    ... )
    >>> pipeline = NanoBananaPipeline(config)
    >>> result = await pipeline.arun(input_data)
"""

from .config import PipelineConfig
from .pipeline import NanoBananaPipeline, run_nano_banana_pipeline
from .types import (
    DocumentContentResult,
    GeneratedImageResult,
    ImagePromptResult,
    MultiHopQuestionResult,
    PartialPipelineOutput,
    PipelineInput,
    PipelineOutput,
    SearchResult,
    SingleHopResult,
)

__all__ = [
    "DocumentContentResult",
    "GeneratedImageResult",
    "ImagePromptResult",
    "MultiHopQuestionResult",
    "NanoBananaPipeline",
    "PartialPipelineOutput",
    "PipelineConfig",
    "PipelineInput",
    "PipelineOutput",
    "SearchResult",
    "SingleHopResult",
    "run_nano_banana_pipeline",
]
