"""Step 1: Single-hop Q&A generation chain (TEXT ONLY)."""

import json

import anthropic

from ..config import PipelineConfig
from ..prompts import SINGLE_HOP_QA_SYSTEM, SINGLE_HOP_QA_USER, SingleHopQAOutput
from ..types import SingleHopResult

# Beta header for structured outputs
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


class SingleHopQAChain:
    """Chain for generating single-hop Q&A using Anthropic API with structured outputs."""

    def __init__(self, config: PipelineConfig):
        """Initialize the chain with configuration."""
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def invoke(self, visual_description: str) -> SingleHopResult:
        """Generate a single-hop question and answer.

        Args:
            visual_description: Visual description of the document.

        Returns:
            SingleHopResult with question, answer, and reasoning.
        """
        # Format the user message
        user_message = SINGLE_HOP_QA_USER.format(
            visual_description=visual_description,
        )

        # Call Anthropic API with structured output
        response = self.client.beta.messages.create(
            model=self.config.sonnet_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=SINGLE_HOP_QA_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            output_format=SingleHopQAOutput,
            betas=[STRUCTURED_OUTPUTS_BETA],
        )

        # Extract and parse JSON output
        parsed = json.loads(response.content[0].text)

        return SingleHopResult(
            question=parsed["question"],
            answer=parsed["answer"],
            reasoning=parsed["reasoning"],
        )

    async def ainvoke(self, visual_description: str) -> SingleHopResult:
        """Async version of invoke."""
        import asyncio

        return await asyncio.to_thread(self.invoke, visual_description)
