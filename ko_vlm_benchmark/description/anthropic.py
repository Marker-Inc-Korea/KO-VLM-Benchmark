import asyncio
from pathlib import Path

from ko_vlm_benchmark.anthropic import claude_multimodal_acomplete


async def generate_description_for_row(
    row_data: dict,
    api_key: str,
    image_base_path: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate rich Korean description for a single row using Claude."""
    # Extract data from row
    row_id = row_data["id"]
    doc_type = row_data["doc_type"]
    visual_context = row_data["visual_context"]
    modified_visual_context = row_data["modified_visual_context"]
    original_image_path = row_data["Orig_image"]
    document = row_data["document"]

    # Construct full image path
    full_image_path = str(image_base_path / doc_type / original_image_path)

    # Build prompt
    instruction = (
        "당신은 주어진 이미지를 기반으로, 풍부한 한국어 (korean) description을 생성하는 어시스턴트 입니다.\n"
        "주어진 설명문과 이미지를 기반으로, 정보가 풍부한 한국어 description을 만들어주세요.\n"
        "설명문에 주어진 수치나 표현들을 참고하여 작성하여야 하며, 이미지에 표현된 도식을 설명하는 문장이 포함하도록 만들어야 합니다.\n"
        "생성된 한국어 description은 주어진 설명에 있는 [차트, 도식, 표]와 1대1 매칭을 시켜야하며, 설명문만 출력해주세요."
    )

    query = f"{instruction}\n설명문:\n{modified_visual_context}\n이미지에 대한 풍부한 description:"

    # Generate with semaphore protection
    async with semaphore:
        try:
            response = await claude_multimodal_acomplete(
                api_key=api_key,
                image_path_list=[full_image_path],
                user_text=query,
            )

            if response:
                # Clean unicode bugs if any
                cleaned_response = response.replace("\u0304", "")
                return {
                    "id": row_id,
                    "doc_type": doc_type,
                    "visual_context": visual_context,
                    "modified_visual_context": modified_visual_context,
                    "Orig_image": original_image_path,
                    "document": document,
                    "Anthropic_GT_1": cleaned_response,
                    "status": "success",
                }
            else:
                return {
                    "id": row_id,
                    "doc_type": doc_type,
                    "visual_context": visual_context,
                    "modified_visual_context": modified_visual_context,
                    "Orig_image": original_image_path,
                    "document": document,
                    "Anthropic_GT_1": None,
                    "status": "failed_empty_response",
                }
        except Exception as e:
            return {
                "id": row_id,
                "doc_type": doc_type,
                "visual_context": visual_context,
                "modified_visual_context": modified_visual_context,
                "Orig_image": original_image_path,
                "document": document,
                "Anthropic_GT_1": None,
                "status": f"error: {e!s}",
            }
