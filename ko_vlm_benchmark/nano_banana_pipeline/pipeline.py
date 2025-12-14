"""Main pipeline orchestration for multi-hop multi-page VQA generation."""

import asyncio
from pathlib import Path

from .chains import (
    DocumentContentChain,
    ImageGenerationChain,
    ImagePromptChain,
    MultiHopQuestionChain,
    SingleHopQAChain,
)
from .config import PipelineConfig
from .types import PartialPipelineOutput, PipelineInput, PipelineOutput


class NanoBananaPipeline:
    """Multi-hop multi-page VQA pipeline using LangChain and LCEL.

    This pipeline generates:
    1. Single-hop Q&A from visual description (text only)
    2. Multi-hop question using web search
    3. Hypothetical document content using web search
    4. Image generation prompt
    5. Generated document image using Gemini (can be skipped for batch processing)

    Example:
        >>> config = PipelineConfig()
        >>> pipeline = NanoBananaPipeline(config)
        >>> result = await pipeline.arun({
        ...     "image_path": "path/to/document.png",
        ...     "visual_description": "A financial report showing...",
        ... })

    Batch mode (skip image generation):
        >>> config = PipelineConfig(skip_image_generation=True)
        >>> pipeline = NanoBananaPipeline(config)
        >>> partial_result = await pipeline.arun(input_data)  # Returns PartialPipelineOutput
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pipeline.

        Args:
            config: Pipeline configuration. If None, uses default config
                    with API keys from environment variables.
        """
        self.config = config or PipelineConfig()
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Build all pipeline chains."""
        self.single_hop_chain = SingleHopQAChain(self.config)
        self.multi_hop_chain = MultiHopQuestionChain(self.config)
        self.document_chain = DocumentContentChain(self.config)
        self.image_prompt_chain = ImagePromptChain(self.config)

        # Only build image generation chain if not skipping
        if not self.config.skip_image_generation:
            self.image_gen_chain = ImageGenerationChain(self.config)

    async def _run_steps_1_to_4(self, input_data: PipelineInput) -> PartialPipelineOutput:
        """Run Steps 1-4 of the pipeline (without image generation).

        Args:
            input_data: Pipeline input.

        Returns:
            Partial pipeline output with Steps 1-4 results.
        """
        # Step 1: Generate single-hop Q&A (text only)
        single_hop_result = await self.single_hop_chain.ainvoke(input_data["visual_description"])

        # Step 2: Generate multi-hop question AND answer with web search
        multi_hop_result = await self.multi_hop_chain.ainvoke(
            single_hop_result=single_hop_result,
            visual_description=input_data["visual_description"],
        )

        # Step 3: Format document content (NO web search - uses Step 2 results)
        document_result = await self.document_chain.ainvoke(
            multi_hop_question=multi_hop_result["multi_hop_question"],
            multi_hop_answer=multi_hop_result["multi_hop_answer"],
            additional_info=multi_hop_result["additional_info"],
            visual_description=input_data["visual_description"],
        )

        # Step 4: Create image generation prompt (text only)
        image_prompt_result = await self.image_prompt_chain.ainvoke(
            visual_description=input_data["visual_description"],
            document_content=document_result["document_content"],
        )

        return PartialPipelineOutput(
            original_image_path=str(input_data["image_path"]),
            original_visual_description=input_data["visual_description"],
            single_hop_question=single_hop_result["question"],
            single_hop_answer=single_hop_result["answer"],
            single_hop_reasoning=single_hop_result["reasoning"],
            multi_hop_question=multi_hop_result["multi_hop_question"],
            multi_hop_answer=multi_hop_result["multi_hop_answer"],
            additional_info=multi_hop_result["additional_info"],
            multi_hop_question_style=multi_hop_result["question_style"],
            search_queries=multi_hop_result["search_queries"],
            search_results=multi_hop_result["search_results"],
            thinking_trajectory=multi_hop_result["thinking_trajectory"],
            hypothetical_document_content=document_result["document_content"],
            image_generation_prompt=image_prompt_result["image_prompt"],
            style_description=image_prompt_result["style_description"],
        )

    async def arun(self, input_data: PipelineInput) -> PipelineOutput | PartialPipelineOutput:
        """Run the pipeline asynchronously.

        If skip_image_generation is True, returns PartialPipelineOutput.
        Otherwise, returns full PipelineOutput including generated image.

        Args:
            input_data: Pipeline input with image path, visual description,
                       and document type.

        Returns:
            Pipeline output (partial or full depending on configuration).
        """
        # Run Steps 1-4
        partial_result = await self._run_steps_1_to_4(input_data)

        # If skipping image generation, return partial result
        if self.config.skip_image_generation:
            return partial_result

        # Step 5: Generate document image (original image + prompt)
        generated_image = await self.image_gen_chain.ainvoke(
            original_image_path=input_data["image_path"],
            image_prompt=partial_result["image_generation_prompt"],
        )

        # Assemble final output
        return PipelineOutput(
            **partial_result,
            generated_image_bytes=generated_image["image_bytes"],
            generated_image_path=generated_image.get("image_path", ""),
        )

    def run(self, input_data: PipelineInput) -> PipelineOutput | PartialPipelineOutput:
        """Synchronous wrapper for arun.

        Args:
            input_data: Pipeline input.

        Returns:
            Pipeline output (partial or full depending on configuration).
        """
        return asyncio.run(self.arun(input_data))


async def run_nano_banana_pipeline(
    image_path: str | Path,
    visual_description: str,
    config: PipelineConfig | None = None,
) -> PipelineOutput | PartialPipelineOutput:
    """Run the nano banana pipeline.

    Convenience function for running the pipeline with minimal setup.

    Args:
        image_path: Path to the original document image.
        visual_description: Visual description of the document.
        config: Optional pipeline configuration.

    Returns:
        Pipeline output (partial or full depending on configuration).

    Example:
        >>> result = await run_nano_banana_pipeline(
        ...     image_path="document.png",
        ...     visual_description="A financial report showing Q3 2024 revenue...",
        ... )
        >>> print(result["multi_hop_question"])
    """
    pipeline = NanoBananaPipeline(config)
    return await pipeline.arun(
        PipelineInput(
            image_path=str(image_path),
            visual_description=visual_description,
        )
    )
