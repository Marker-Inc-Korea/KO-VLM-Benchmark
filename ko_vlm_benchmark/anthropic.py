import asyncio
import base64
import os
from typing import Any

import aiofiles
import aiohttp

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def get_media_type(file_path: str) -> str:
    """Detect media type from file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_type_map.get(ext, "image/jpeg")


async def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and return with its media type."""
    async with aiofiles.open(image_path, "rb") as f:
        content = await f.read()
    base64_data = base64.b64encode(content).decode("utf-8")
    media_type = get_media_type(image_path)
    return base64_data, media_type


async def send_multimodal_request(
    session: aiohttp.ClientSession,
    api_key: str,
    image_data_list: list[tuple[str, str]],
    user_text: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 1024,
) -> dict[str, Any]:
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}

    # Build content with multiple images, each with its own media type
    content = []
    for image_base64, media_type in image_data_list:
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_base64}})

    # Add text after all images
    content.append({"type": "text", "text": user_text})

    data = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}

    async with session.post(ANTHROPIC_URL, json=data, headers=headers) as resp:
        return await resp.json()


async def claude_multimodal_acomplete(
    api_key: str,
    image_path_list: list[str],
    user_text: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 1024,
) -> str | None:
    # Encode all images concurrently
    image_base64_list = await asyncio.gather(*[encode_image_to_base64(image_path) for image_path in image_path_list])

    # Send a single request with all images
    async with aiohttp.ClientSession() as session:
        result = await send_multimodal_request(
            session, api_key, image_base64_list, user_text, model=model, max_tokens=max_tokens
        )

    # Extract text from response
    if "content" in result and len(result["content"]) > 0:
        return result["content"][0]["text"]
    else:
        return None
