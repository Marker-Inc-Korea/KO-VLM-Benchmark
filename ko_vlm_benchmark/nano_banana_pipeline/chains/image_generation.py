"""Step 5: Image generation chain using Gemini (Nano Banana Pro)."""

import uuid
from pathlib import Path

from google import genai
from google.genai import types

from ..config import PipelineConfig
from ..types import GeneratedImageResult


def get_mime_type(image_path: str | Path) -> str:
    """Get MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


def build_full_prompt(image_prompt: str) -> str:
    """Build the full prompt with style reference instructions."""
    return (
        f"제공된 참조 이미지의 시각적 스타일을 따라 새로운 문서 이미지를 생성하세요. "
        f"새 문서는 다음을 포함해야 합니다:\n\n"
        f"{image_prompt}\n\n"
        f"중요: 참조 이미지의 레이아웃, 타이포그래피, 색상 구성 및 전체적인 "
        f"미학을 맞추되, 참조 이미지와는 완전히 다른 내용으로 작성해야 합니다."
    )


def build_batch_request(
    client: genai.Client,
    original_image_path: str | Path,
    image_prompt: str,
) -> dict:
    """Build a Gemini batch API request for image generation.

    Args:
        original_image_path: Path to the original document image (for style).
        image_prompt: Detailed prompt describing the document to generate.

    Returns:
        Request dict for Gemini batch API.
    """
    image_file = client.files.upload(file=original_image_path)
    full_prompt = build_full_prompt(image_prompt)

    return {
        "contents": [
            {
                "parts": [
                    {"text": full_prompt},
                    {"file_data": {"file_uri": image_file.uri, "mime_type": image_file.mime_type}},
                ]
            }
        ]
    }


class ImageGenerationChain:
    """Chain for generating document images using Gemini."""

    def __init__(self, config: PipelineConfig):
        """Initialize the chain with configuration."""
        self.config = config
        self.client = genai.Client(api_key=config.google_api_key)

    def _save_image(self, image_bytes: bytes) -> str:
        """Save generated image to output directory and return path."""
        output_dir = self.config.ensure_output_dir()
        filename = f"generated_{uuid.uuid4().hex[:8]}.png"
        output_path = output_dir / filename

        with open(output_path, "wb") as f:
            f.write(image_bytes)

        return str(output_path)

    def invoke(
        self,
        original_image_path: str | Path,
        image_prompt: str,
    ) -> GeneratedImageResult:
        """Generate a document image using Gemini.

        The original image is essential for Nano Banana Pro to follow
        the visual style of the original document.

        Args:
            original_image_path: Path to the original document image (for style).
            image_prompt: Detailed prompt describing the document to generate.

        Returns:
            GeneratedImageResult with image bytes and saved path.
        """
        # Read original image
        with open(original_image_path, "rb") as f:
            original_image_bytes = f.read()

        mime_type = get_mime_type(original_image_path)
        full_prompt = build_full_prompt(image_prompt)

        # Call Gemini API with original image for style reference
        response = self.client.models.generate_content(
            model=self.config.gemini_image_model,
            contents=[
                types.Part.from_bytes(data=original_image_bytes, mime_type=mime_type),
                full_prompt,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        # Extract generated image from response
        image_bytes: bytes | None = None
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.inline_data is not None:
                        image_bytes = part.inline_data.data
                        break

        if image_bytes is None:
            raise ValueError("No image generated in response")

        # Save the image
        saved_path = self._save_image(image_bytes)

        return GeneratedImageResult(
            image_bytes=image_bytes,
            image_path=saved_path,
        )

    async def ainvoke(
        self,
        original_image_path: str | Path,
        image_prompt: str,
    ) -> GeneratedImageResult:
        """Async version of invoke."""
        import asyncio

        return await asyncio.to_thread(self.invoke, original_image_path, image_prompt)
