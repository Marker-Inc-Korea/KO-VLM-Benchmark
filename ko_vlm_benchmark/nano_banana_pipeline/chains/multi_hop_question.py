"""Step 2: Multi-hop question generation chain with Anthropic web search."""

import json
import secrets

import anthropic

from ..config import PipelineConfig
from ..prompts import (
    MULTI_HOP_QUESTION_SYSTEM,
    MULTI_HOP_QUESTION_USER,
    STYLE_INPUT_LIST,
    MultiHopQuestionOutput,
)
from ..types import MultiHopQuestionResult, SingleHopResult
from .util import _get_last_text_block_content

# Beta header for structured outputs
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


class MultiHopQuestionChain:
    """Chain for generating multi-hop questions using Anthropic's web search tool."""

    def __init__(self, config: PipelineConfig):
        """Initialize the chain with configuration."""
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def invoke(
        self,
        single_hop_result: SingleHopResult,
        visual_description: str,
    ) -> MultiHopQuestionResult:
        """Generate a multi-hop question using web search.

        Args:
            single_hop_result: Result from Step 1 (single-hop Q&A).
            visual_description: Original visual description of the document.

        Returns:
            MultiHopQuestionResult with the generated question and search results.
        """
        question_style = secrets.choice(STYLE_INPUT_LIST)

        # Format the user message
        user_message = MULTI_HOP_QUESTION_USER.format(
            single_hop_question=single_hop_result["question"],
            single_hop_answer=single_hop_result["answer"],
            visual_description=visual_description,
            style_input=question_style,
        )

        response = self.client.beta.messages.create(
            model=self.config.sonnet_model,
            max_tokens=self.config.max_tokens,
            system=MULTI_HOP_QUESTION_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            output_format=MultiHopQuestionOutput,
            betas=[STRUCTURED_OUTPUTS_BETA],
        )

        # Extract parsed output
        parsed = json.loads(_get_last_text_block_content(response))

        return MultiHopQuestionResult(
            multi_hop_question=parsed["multi_hop_question"],
            additional_info_needed=parsed["additional_info_needed"],
            question_style=question_style,
        )

    async def ainvoke(
        self,
        single_hop_result: SingleHopResult,
        visual_description: str,
    ) -> MultiHopQuestionResult:
        """Async version of invoke."""
        import asyncio

        return await asyncio.to_thread(self.invoke, single_hop_result, visual_description)
