## multi-choice question make


###################################### not async
# from ko_vlm_benchmark.anthropic import claude_multimodal_acomplete
import base64
import os
from typing import Any

import requests

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


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and return with its media type."""
    with open(image_path, "rb") as f:
        content = f.read()
    base64_data = base64.b64encode(content).decode("utf-8")
    media_type = get_media_type(image_path)
    return base64_data, media_type


def send_multimodal_request(
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

    response = requests.post(ANTHROPIC_URL, json=data, headers=headers, timeout=300)
    return response.json()


def claude_multimodal_acomplete(
    api_key: str,
    image_path_list: list[str],
    user_text: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 1024,
) -> str | None:
    # Encode all images concurrently
    image_base64_list = [encode_image_to_base64(image_path) for image_path in image_path_list]

    # Send a single request with all images
    result = send_multimodal_request(api_key, image_base64_list, user_text, model=model, max_tokens=max_tokens)

    # Extract text from response
    if "content" in result and len(result["content"]) > 0:
        return result["content"][0]["text"]
    else:
        return None


######################################


# nemerical만 부분만 거의 다른 wrong answer
def generate_wrong_answer_numerical(
    image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
):

    # find proper prompt
    instruction = (
        "당신은 주어진 [images, question, right answer]을 기반으로, 틀린 답변을 생성해내는 어시스턴트입니다.\n"
        "주어진 images와 question을 보고 wrong answer을 생성할 때, right answer가 가지고 있는 전체적인 문장 정보는 동일하게 유지하면서 **수치적인 정보**만 살짝 다르게 만들어야 합니다. \n"
        "이렇게 생성된 wrong answer는 right answer와 **수치적인 부분만 다르며**, AI나 사람이 맞추기 힘든 hard-case여야 합니다.\n"
        "생성된 wrong answer는 반드시 한국어로 출력해야하며, 주어진 정보들을 기반으로 명령을 잘 수행하세요."
    )

    query = f"{instruction}\nQuestion: {question}\nAnswer: {gt_answer}\n수치적인 부분만 달라진 wrong answer:"
    print(query)

    # need api_key
    response = claude_multimodal_acomplete(
        api_key=api_key,
        image_path_list=[image_path1, image_path2],
        user_text=query,
    )
    print(response)

    if response:
        # Clean unicode bugs if any
        cleaned_response = response.replace("\u0304", "")

        return cleaned_response

    else:
        return None


def generate_wrong_answer_meaning():
    pass


def generate_wrong_answer_order():
    pass
