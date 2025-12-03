from typing import Any

from ..types import SearchResult


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
