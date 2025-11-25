import copy

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from ko_vlm_benchmark.anthropic import claude_multimodal_acomplete
from ko_vlm_benchmark.exceptions import NullResponseError

two_hop_incremental_prompt = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content="주어진 모든 문서를 참조해야만 하는, 주어진 답변에 대한 다단계(multi-hop) 질문을 생성하세요. 생성한 질문의 정답 답변은 주어진 답변과 일치해야 하고, 두 문서를 모두 참조해야만 합니다.",
    ),
    ChatMessage(
        role=MessageRole.USER,
        content="""문서 1: 누에보 라레도(Nuevo Laredo) 시는 멕시코 타마울리파스(Tamaulipas) 주에 위치합니다.
문서 2: 시우다드 데포르티바(Ciudad Deportiva, 스포츠 시티)는 멕시코 누에보 라레도의 스포츠 복합 단지입니다. 이곳은 멕시코 야구 리그 팀인 테콜로테스 데누에보 라레도(Tecolotes de Nuevo Laredo)의 홈구장이며 ...""",
    ),
    ChatMessage(
        role=MessageRole.ASSISTANT,
        content="""답변: 타마울리파스(Tamaulipas) 주
1단계 질문 (문서 1 사용): 누에보 라레도(Nuevo Laredo)는 멕시코 어느 주에 위치해 있나요?
2단계 질문 (문서 2 사용): 테콜로테스 데 누에보 라레도(Tecolotes de Nuevo Laredo)의 홈구장인 시우다드 데포르티바(Ciudad Deportiva)는 멕시코 어느 주에서 찾을 수 있나요?""",
    ),
]


async def generate_original_query(
    document: str,
    llm: BaseLLM,
) -> str:
    """
    Generate an original query from a single document.
    This query will be used as the first-hop question in the two-hop generation process.
    """
    prompt = f"""주어진 문서의 내용을 바탕으로 간단하고 명확한 질문 하나를 생성하세요.
질문은 문서에서 직접 답변할 수 있는 내용이어야 합니다.

문서: {document}

질문:"""

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="당신은 문서 기반 질문을 생성하는 AI 어시스턴트입니다."),
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]

    chat_response = await llm.achat(messages)
    response = chat_response.message.content
    if response is None:
        return ""
    return response.strip()


async def generate_desired_answer(
    original_query: str,
    document: str,
    image_path: str,
    api_key: str,
) -> str:
    """
    Generate a desired answer for the original query using MultiModalLLM.
    Uses both the visual context description and the actual image.
    """
    prompt = f"""주어진 질문에 대해 문서와 이미지를 참고하여 정확하고 간결한 답변을 생성하세요.

질문: {original_query}

문서 설명: {document}

답변:"""

    # Use MultiModalLLM to generate answer with both text and image
    response = await claude_multimodal_acomplete(api_key, [image_path], prompt)
    if response is None:
        raise NullResponseError
    return response


async def generate_two_hop_question(
    desired_answer: str,
    original_document: str,
    added_document: str,
    original_query: str,
    llm: BaseLLM,
) -> str:
    context_str = f"문서 1: {original_document}\n문서 2: {added_document}"
    assistant_prompt = f"답변: {desired_answer}\n1단계 질문 (문서 1 사용): {original_query}\n2단계 질문 (문서 2 사용):"
    messages = copy.deepcopy(two_hop_incremental_prompt)
    messages.extend([
        ChatMessage(role=MessageRole.USER, content=context_str),
        ChatMessage(role=MessageRole.ASSISTANT, content=assistant_prompt),
    ])

    chat_response = await llm.achat(messages)
    response = chat_response.message.content
    if response is None:
        raise NullResponseError
    return response.split(":")[-1].strip()


async def generate_two_hop_answer(
    multi_hop_question: str,
    documents: list[str],
    llm: BaseLLM,
) -> str:
    context_str = "\n\n".join([f"문서 {i + 1}: {doc}" for i, doc in enumerate(documents)])
    prompt = (
        "주어진 모든 문서를 참조해야만 하는, 주어진 답변에 대한 다단계(multi-hop) 질문에 대해 정확한 답변을 생성하세요."
    )
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt),
        ChatMessage(role=MessageRole.USER, content=f"{context_str}\n\n질문: {multi_hop_question}\n\n답변:"),
    ]

    chat_response = await llm.achat(messages)
    response = chat_response.message.content
    if response is None:
        raise NullResponseError
    return response.strip()


async def generate_model_answer(
    question: str,
    image_paths: list[str],
    api_key: str,
    model: str = "claude-opus-4-5-20251101",
    max_tokens: int = 2048,
) -> str:
    """
    Generate a model answer for a given question using two document images.

    Args:
        question: The multi-hop question to answer.
        image_paths: List of paths to image files.
        api_key: Anthropic API key.
        model: Model to use for answer generation.
        max_tokens: Maximum tokens for the response.

    Returns:
        The generated model answer.
    """
    prompt = f"""다음은 두 개의 문서 이미지입니다. 주어진 질문에 대해 두 문서를 모두 참고하여 정확하고 상세한 답변을 생성하세요.

질문: {question}

답변:"""

    response = await claude_multimodal_acomplete(
        api_key=api_key,
        image_path_list=image_paths,
        user_text=prompt,
        model=model,
        max_tokens=max_tokens,
    )

    if response is None:
        raise NullResponseError

    return response
