"""Step 2: Multi-hop question generation chain with Anthropic web search."""

import json
import secrets
from typing import Any

import anthropic

from ..config import PipelineConfig
from ..prompts import (
    MULTI_HOP_QUESTION_SYSTEM,
    MULTI_HOP_QUESTION_USER,
    STYLE_INPUT_LIST,
    MultiHopQuestionOutput,
)
from ..types import MultiHopQuestionResult, SingleHopResult
from .util import (
    _extract_search_queries,
    _extract_search_results,
    _extract_thinking_contents,
    _get_last_text_block_content,
)

# Beta header for structured outputs
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


class MultiHopQuestionChain:
    """Chain for generating multi-hop questions and answers using Anthropic's web search tool."""

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
        """Generate a multi-hop question and answer using web search.

        Args:
            single_hop_result: Result from Step 1 (single-hop Q&A).
            visual_description: Original visual description of the document.

        Returns:
            MultiHopQuestionResult with the generated question, answer, and search results.
        """
        question_style = secrets.choice(STYLE_INPUT_LIST)

        # Format the user message
        user_message = MULTI_HOP_QUESTION_USER.format(
            single_hop_question=single_hop_result["question"],
            single_hop_answer=single_hop_result["answer"],
            visual_description=visual_description,
            style_input=question_style,
        )

        # Call Anthropic API with web search tool
        response = self.client.beta.messages.create(
            model=self.config.sonnet_model,
            max_tokens=self.config.max_tokens,
            system=MULTI_HOP_QUESTION_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
            output_format=MultiHopQuestionOutput,
            betas=[STRUCTURED_OUTPUTS_BETA],
            tools=self._build_tools(),
            thinking={"type": "enabled", "budget_tokens": 10000},
        )

        # Extract search queries and results from the response
        search_queries = _extract_search_queries(response)
        search_results = _extract_search_results(response)
        thinking_trajectory = _extract_thinking_contents(response)

        # Extract parsed output from the last text block
        parsed = json.loads(_get_last_text_block_content(response))

        return MultiHopQuestionResult(
            multi_hop_question=parsed["multi_hop_question"],
            multi_hop_answer=parsed["multi_hop_answer"],
            additional_info=parsed["additional_info"],
            question_style=question_style,
            search_queries=search_queries,
            search_results=search_results,
            thinking_trajectory=thinking_trajectory,
        )

    async def ainvoke(
        self,
        single_hop_result: SingleHopResult,
        visual_description: str,
    ) -> MultiHopQuestionResult:
        """Async version of invoke."""
        import asyncio

        return await asyncio.to_thread(self.invoke, single_hop_result, visual_description)
