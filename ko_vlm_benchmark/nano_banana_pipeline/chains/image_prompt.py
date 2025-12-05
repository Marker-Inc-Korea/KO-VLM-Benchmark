"""Step 4: Image generation prompt creation chain (TEXT ONLY)."""

import json

import anthropic

from ..config import PipelineConfig
from ..prompts import IMAGE_PROMPT_SYSTEM, IMAGE_PROMPT_USER, ImagePromptOutput
from ..types import ImagePromptResult

# Beta header for structured outputs
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


class ImagePromptChain:
    """Chain for generating image prompts using Anthropic API with structured outputs."""

    def __init__(self, config: PipelineConfig):
        """Initialize the chain with configuration."""
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def invoke(
        self,
        visual_description: str,
        document_content: str,
    ) -> ImagePromptResult:
        """Generate an image generation prompt.

        Args:
            visual_description: Visual description of the original document.
            document_content: Generated document content.

        Returns:
            ImagePromptResult with the generated prompt and style description.
        """
        # Format the user message
        user_message = IMAGE_PROMPT_USER.format(
            visual_description=visual_description,
            document_content=document_content,
        )

        # Call Anthropic API with structured output
        response = self.client.beta.messages.create(
            model=self.config.sonnet_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=IMAGE_PROMPT_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            output_format=ImagePromptOutput,
            betas=[STRUCTURED_OUTPUTS_BETA],
        )

        # Extract parsed output
        parsed = json.loads(response.content[0].text)

        return ImagePromptResult(
            image_prompt=parsed["image_prompt"],
            style_description=parsed["style_description"],
        )

    async def ainvoke(
        self,
        visual_description: str,
        document_content: str,
    ) -> ImagePromptResult:
        """Async version of invoke."""
        import asyncio

        return await asyncio.to_thread(self.invoke, visual_description, document_content)
