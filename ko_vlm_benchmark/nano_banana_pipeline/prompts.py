"""Prompt templates and structured output schemas for the nano banana pipeline."""


# =============================================================================
# Structured Output Schemas (Pydantic Models)
# =============================================================================

SingleHopQAOutput = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"}, "reasoning": {"type": "string"}},
        "required": ["question", "answer", "reasoning"],
        "additionalProperties": False,
    },
}


MultiHopQuestionOutput = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "multi_hop_question": {"type": "string"},
            "additional_info_needed": {"type": "string"},
        },
        "required": ["multi_hop_question", "additional_info_needed"],
        "additionalProperties": False,
    },
}


ImagePromptOutput = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "style_description": {"type": "string"},
            "image_prompt": {"type": "string"},
        },
        "required": ["style_description", "image_prompt"],
        "additionalProperties": False,
    },
}

# =============================================================================
# Step 1: Single-hop Q&A Generation (TEXT ONLY)
# =============================================================================

SINGLE_HOP_QA_SYSTEM = """당신은 시각적 문서 설명을 분석하여 단일 단계(single-hop) 질문과 답변을 생성하는 AI입니다.

주어진 문서 이미지 설명을 바탕으로:
1. 설명에서 직접 답변할 수 있는 명확한 질문을 생성하세요.
2. 해당 질문에 대한 정확한 답변을 제공하세요.
3. 답변에 대한 근거를 설명하세요."""

SINGLE_HOP_QA_USER = """문서 이미지 설명:
{visual_description}

위 설명을 바탕으로 single-hop 질문과 답변을 생성해주세요."""

# =============================================================================
# Step 2: Multi-hop Question Generation (with web_search)
# =============================================================================

MULTI_HOP_QUESTION_SYSTEM = """당신은 다단계(multi-hop) 질문을 생성하는 AI입니다.

주어진 single-hop 질문과 답변을 바탕으로, 추가 정보가 필요한 multi-hop 질문을 생성합니다.

multi-hop 질문의 조건:
1. 원본 문서의 정보(visual_description)를 필요로 합니다.
2. 추가적인 외부 정보도 필요로 합니다. 이 때, 추가적인 외부 정보는 그 어떤 것이던 이미 존재한다고 가정합니다.
3. 두 정보를 종합해야만 완전한 답변이 가능해야 합니다.
4. 두 문서 중 하나라도 없어지면 답할 수 없는 질문이어야 하며, 당연히 질문 내에 답변은 포함되지 않아야 합니다.

Multi-hop 질문을 답하기 위한 정보가 모두 존재한다고 가정하고, multi-hop 질문을 생성하세요.
Multi-hop 질문을 생성한 이후에는, 기존의 주어진 문서 외에 multi-hop 질문에 답하기 위해 필요한 "추가 정보"가 무엇인지도 명시하세요.
"""

MULTI_HOP_QUESTION_USER = """원본 질문: {single_hop_question}
원본 답변: {single_hop_answer}
문서 설명: {visual_description}

위를 바탕으로 multi-hop 질문을 생성한 후, multi-hop 질문에 답하기 위해 필요한 "추가 정보"도 명시해주세요."""

# =============================================================================
# Step 3: Document Content Generation (with web_search for additional info)
# =============================================================================

DOCUMENT_CONTENT_SYSTEM = """당신은 가상의 문서 내용을 생성하는 AI입니다.

multi-hop 질문에 답하기 위해 필요한 "추가 정보"를 담은 새로운 문서를 생성합니다.

중요한 규칙:
1. 원본 문서 설명(visual_description)의 정보와 완전히 독립적인 새로운 내용이어야 합니다.
2. multi-hop 질문에 답하는 데 필요한 추가 정보를 참고하여, 해당 추가 정보와 관련한 내용까지 더불어서 생성합니다.
3. 질문에 대한 답변을 생성하는 것이 아닙니다. 요청한 추가 정보에 대한 가상의 문서를 아주 자세하게 작성합니다. 자세한 정보가 포함될수록 좋습니다.
4. 표, 차트, 그래프, 통계 등 시각적으로 볼 수 있는 정보들이 많을 수록 좋습니다.
5. 웹 검색 도구를 사용하여 관련 정보를 조사하고, 해당 내용들을 문서에 포함하세요.

웹 검색 도구를 사용하여 "필요한 추가 정보"에 대한 실제 데이터를 찾고,
이를 바탕으로 가상의 문서 내용을 생성하세요."""

DOCUMENT_CONTENT_USER = """원본 문서: {visual_description}
Multi-hop 질문: {multi_hop_question}
필요한 추가 정보: {additional_info_needed}

위 정보를 바탕으로 웹 검색을 수행하고, 가상의 문서 내용을 생성해주세요.
이 문서는 원본 문서와 독립적이면서, multi-hop 질문에 답하기 위해 필요한 추가 정보를 담고 있어야 합니다."""

# =============================================================================
# Step 4: Image Generation Prompt Creation (TEXT ONLY)
# =============================================================================

IMAGE_PROMPT_SYSTEM = """당신은 문서 이미지 생성을 위한 상세한 프롬프트를 작성하는 AI입니다.

주어진 문서 내용을 바탕으로,
Gemini 이미지 생성 모델을 위한 상세한 한국어 프롬프트를 작성합니다.

프롬프트 작성 지침:
1. 원본 문서의 시각적 스타일(레이아웃, 색상, 폰트 스타일, 문서 형식 등)을 설명해야 합니다.
2. 생성할 문서의 구체적인 내용을 포함해야 합니다.
3. 표, 차트, 그래프 등 복잡한 구조를 명시해야 합니다.
4. 전문적이고 공식적인 문서 스타일을 유지해야 합니다.
5. 프롬프트는 반드시 한국어로 작성해야 합니다."""

IMAGE_PROMPT_USER = """원본 문서 설명: {visual_description}

생성할 문서 내용:
{document_content}

위 정보를 바탕으로 이미지 생성 프롬프트를 작성해주세요.
프롬프트는 한국어로 작성하고, 이미지 생성 모델이 이해할 수 있도록 구체적으로 작성해주세요."""
