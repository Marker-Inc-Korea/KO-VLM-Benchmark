"""Step 2: Multi-hop question generation chain with Anthropic web search."""

import json
from typing import Any

import anthropic

from ..config import PipelineConfig
from ..prompts import (
    MULTI_HOP_QUESTION_SYSTEM,
    MULTI_HOP_QUESTION_USER,
    MultiHopQuestionOutput,
)
from ..types import MultiHopQuestionResult, SingleHopResult
from .util import _extract_search_results, _get_last_text_block_content

# Beta header for structured outputs
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


class MultiHopQuestionChain:
    """Chain for generating multi-hop questions using Anthropic's web search tool."""

    def __init__(self, config: PipelineConfig):
        """Initialize the chain with configuration."""
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build the web search tool configuration."""
        tool: dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": self.config.web_search_max_uses,
        }

        if self.config.web_search_allowed_domains:
            tool["allowed_domains"] = self.config.web_search_allowed_domains

        return [tool]

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
        # Format the user message
        user_message = MULTI_HOP_QUESTION_USER.format(
            single_hop_question=single_hop_result["question"],
            single_hop_answer=single_hop_result["answer"],
            visual_description=visual_description,
        )

        # Call Anthropic API with web search tool and structured output
        response = self.client.beta.messages.create(
            model=self.config.sonnet_model,
            max_tokens=self.config.max_tokens,
            system=MULTI_HOP_QUESTION_SYSTEM,
            tools=self._build_tools(),
            messages=[{"role": "user", "content": user_message}],
            output_format=MultiHopQuestionOutput,
            betas=[STRUCTURED_OUTPUTS_BETA],
        )

        # Extract search results from the response
        search_results = _extract_search_results(response)

        # Extract parsed output
        parsed = json.loads(_get_last_text_block_content(response))

        return MultiHopQuestionResult(
            multi_hop_question=parsed["multi_hop_question"],
            additional_info_needed=parsed["additional_info_needed"],
            search_results=search_results,
        )

    async def ainvoke(
        self,
        single_hop_result: SingleHopResult,
        visual_description: str,
    ) -> MultiHopQuestionResult:
        """Async version of invoke."""
        import asyncio

        return await asyncio.to_thread(self.invoke, single_hop_result, visual_description)
