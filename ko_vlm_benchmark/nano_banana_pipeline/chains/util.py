from typing import Any

from ..types import SearchResult


def _extract_search_queries(response: Any) -> list[str]:
    """Extract web search queries from the response content blocks.

    Args:
        response: Anthropic API response object.

    Returns:
        List of search query strings used during the API call.
    """
    queries: list[str] = []
    content = getattr(response, "content", [])

    for block in content:
        if getattr(block, "name", None) == "web_search":
            input_data = getattr(block, "input", {})
            query = input_data.get("query", "")
            if query:
                queries.append(query)

    return queries


def _extract_search_results(response: Any) -> list[SearchResult]:
    """Extract search results from the response content blocks.

    Works with both standard Message and beta ParsedMessage responses.
    """
    results: list[SearchResult] = []

    # Handle both standard and beta response types
    content = getattr(response, "content", [])

    for block in content:
        if getattr(block, "type", None) == "web_search_tool_result":
            block_content = getattr(block, "content", [])
            for search_result in block_content:
                if hasattr(search_result, "title"):
                    results.append(
                        SearchResult(
                            title=getattr(search_result, "title", ""),
                            url=getattr(search_result, "url", ""),
                            content=getattr(search_result, "page_content", ""),
                        )
                    )

    return results


def _get_last_text_block_content(response):
    for block in reversed(response.content):
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("No text block in response.content")


def _get_all_text_content(response):
    """모든 text 블록의 내용을 합쳐서 반환."""
    texts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            texts.append(block.text)
    if not texts:
        raise ValueError("No text block in response.content")
    return "\n".join(texts)
