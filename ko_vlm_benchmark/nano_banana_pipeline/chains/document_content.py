"""Step 3: Document content generation chain with Anthropic web search."""

from typing import Any

import anthropic

from ..config import PipelineConfig
from ..prompts import (
    DOCUMENT_CONTENT_SYSTEM,
    DOCUMENT_CONTENT_USER,
)
from ..types import DocumentContentResult
from .util import _extract_search_results, _get_last_text_block_content

# Beta header for structured outputs
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


class DocumentContentChain:
    """Chain for generating document content using Anthropic's web search tool."""

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
        multi_hop_question: str,
        additional_info_needed: str,
        visual_description: str,
    ) -> DocumentContentResult:
        """Generate hypothetical document content using web search.

        This searches for ONLY the additional info needed (not the visual description).
        The visual description is the "grounded passage" that doesn't need searching.

        Args:
            multi_hop_question: The multi-hop question from Step 2.
            additional_info_needed: Description of what additional info is needed.

        Returns:
            DocumentContentResult with the generated document content.
        """
        # Format the user message
        user_message = DOCUMENT_CONTENT_USER.format(
            multi_hop_question=multi_hop_question,
            additional_info_needed=additional_info_needed,
            visual_description=visual_description,
        )

        # Call Anthropic API with web search tool and structured output
        response = self.client.beta.messages.create(
            model=self.config.sonnet_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=DOCUMENT_CONTENT_SYSTEM,
            tools=self._build_tools(),
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract search results from the response
        search_results = _extract_search_results(response)

        # Extract parsed output
        content = _get_last_text_block_content(response)

        return DocumentContentResult(
            document_content=content,
            search_results=search_results,
        )

    async def ainvoke(
        self,
        multi_hop_question: str,
        additional_info_needed: str,
        visual_description: str,
    ) -> DocumentContentResult:
        """Async version of invoke."""
        import asyncio

        return await asyncio.to_thread(self.invoke, multi_hop_question, additional_info_needed, visual_description)
