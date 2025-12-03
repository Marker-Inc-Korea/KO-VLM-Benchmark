"""Type definitions for the nano banana pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from typing_extensions import NotRequired


class SearchResult(TypedDict):
    """Individual web search result."""

    title: str
    url: str
    content: str


class PipelineInput(TypedDict):
    """Input to the pipeline."""

    image_path: str | Path
    visual_description: str


class SingleHopResult(TypedDict):
    """Output from Step 1: Single-hop Q&A."""

    question: str
    answer: str
    reasoning: str


class MultiHopQuestionResult(TypedDict):
    """Output from Step 2: Multi-hop question generation."""

    multi_hop_question: str
    additional_info_needed: str
    search_results: list[SearchResult]


class DocumentContentResult(TypedDict):
    """Output from Step 3: Document content generation."""

    document_content: str
    search_results: list[SearchResult]


class ImagePromptResult(TypedDict):
    """Output from Step 4: Image generation prompt."""

    image_prompt: str
    style_description: str


class GeneratedImageResult(TypedDict):
    """Output from Step 5: Image generation."""

    image_bytes: bytes
    image_path: NotRequired[str]


class PartialPipelineOutput(TypedDict):
    """Pipeline output without image generation (Steps 1-4 only).

    Used when skip_image_generation=True for batch processing.
    """

    # Input preserved
    original_image_path: str
    original_visual_description: str

    # Step 1 outputs
    single_hop_question: str
    single_hop_answer: str
    single_hop_reasoning: str

    # Step 2 outputs
    multi_hop_question: str
    additional_info_needed: str

    # Step 3 outputs
    hypothetical_document_content: str

    # Step 4 outputs
    image_generation_prompt: str
    style_description: str


class PipelineOutput(PartialPipelineOutput):
    """Complete pipeline output including image generation."""

    # Step 5 outputs
    generated_image_bytes: bytes
    generated_image_path: str
