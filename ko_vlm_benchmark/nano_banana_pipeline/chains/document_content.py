"""Step 3: Document content formatting chain (NO web search - uses Step 2 results)."""

import anthropic

from ..config import PipelineConfig
from ..prompts import (
    DOCUMENT_CONTENT_SYSTEM,
    DOCUMENT_CONTENT_USER,
)
from ..types import DocumentContentResult
from .util import _get_all_text_content


class DocumentContentChain:
    """Chain for formatting document content from Step 2's search results."""

    def __init__(self, config: PipelineConfig):
        """Initialize the chain with configuration."""
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def invoke(
        self,
        multi_hop_question: str,
        multi_hop_answer: str,
        additional_info: str,
        visual_description: str,
    ) -> DocumentContentResult:
        """Format document content from Step 2's additional info.

        No web search is performed - this step only formats the information
        already gathered in Step 2 into a document structure.

        Args:
            multi_hop_question: The multi-hop question from Step 2.
            multi_hop_answer: The answer to the multi-hop question from Step 2.
            additional_info: The additional info gathered from web search in Step 2.
            visual_description: Original visual description of the document.

        Returns:
            DocumentContentResult with the formatted document content.
        """
        # Format the user message
        user_message = DOCUMENT_CONTENT_USER.format(
            multi_hop_question=multi_hop_question,
            multi_hop_answer=multi_hop_answer,
            additional_info=additional_info,
            visual_description=visual_description,
        )

        # Call Anthropic API WITHOUT web search tool
        response = self.client.messages.create(
            model=self.config.sonnet_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=DOCUMENT_CONTENT_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract content
        content = _get_all_text_content(response)

        return DocumentContentResult(
            document_content=content,
        )

    async def ainvoke(
        self,
        multi_hop_question: str,
        multi_hop_answer: str,
        additional_info: str,
        visual_description: str,
    ) -> DocumentContentResult:
        """Async version of invoke."""
        import asyncio

        return await asyncio.to_thread(
            self.invoke, multi_hop_question, multi_hop_answer, additional_info, visual_description
        )
