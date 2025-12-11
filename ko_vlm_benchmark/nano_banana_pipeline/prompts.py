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
            "multi_hop_answer": {"type": "string"},
            "additional_info": {"type": "string"},
        },
        "required": ["multi_hop_question", "multi_hop_answer", "additional_info"],
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
# Step 2: Multi-hop Question Generation (with web search)
# =============================================================================

MULTI_HOP_QUESTION_SYSTEM = """당신은 웹 검색을 통해 외부 지식을 수집하고, 이를 활용하여 다단계(multi-hop) 질문과 답변을 생성하는 AI 어시스턴트입니다.

## 작업 순서:
1. 먼저 웹 검색 도구를 사용하여 주어진 문서와 연결될 수 있는 관련 외부 정보를 검색하세요.
2. 검색 결과를 바탕으로 multi-hop 질문, 답변, 그리고 추가 정보를 생성하세요.

## Multi-hop 질문 생성 조건:
1. 주어진 원본 문서의 정보(visual_description)를 기반으로, single-hop 질문이 만들어졌습니다.
2. 웹 검색으로 찾은 실제 외부 정보와 원본 문서를 결합하여 multi-hop 질문을 만듭니다.
3. Multi-hop 질문은 원본 문서의 정보와 검색으로 찾은 외부 정보를 모두 활용해야 답변할 수 있어야 합니다.
4. 생성된 Multi-hop 질문은 원본 문서만으로는 완전한 답변이 불가능해야 합니다.
5. 사용자가 원하는 질문 스타일에 맞는 multi-hop 질문을 생성해야 합니다.

## 출력 형식:
웹 검색을 완료한 후, 다음 JSON 형태로 답변하세요:
{
    "multi_hop_question": "[생성된 multi-hop 질문]",
    "multi_hop_answer": "[웹 검색 결과와 원본 문서를 기반으로 한 정답]",
    "additional_info": "[multi-hop 질문에 답하기 위해 필요한 외부 정보 요약 - 검색에서 찾은 실제 정보]"
}

중요: additional_info는 웹 검색에서 실제로 찾은 정보를 기반으로 작성해야 합니다."""

MULTI_HOP_QUESTION_USER = """원본 질문: {single_hop_question}
원본 답변: {single_hop_answer}
문서 설명: {visual_description}

위를 바탕으로 먼저 웹 검색을 수행하여 관련 외부 정보를 찾은 후,
{style_input}의 multi-hop 질문과 답변을 생성해주세요."""


# Step 2-1: style_input options
STYLE_INPUT_LIST = [
    "분석형 스타일",
    "비교/대조 스타일",
    "추론형 스타일",
    "close-ended 스타일",
]

# =============================================================================
# Step 3: Document Content Formatting (NO web search - uses Step 2 results)
# =============================================================================

DOCUMENT_CONTENT_SYSTEM = """당신은 주어진 정보를 바탕으로 가상의 문서 내용을 포맷팅하는 AI입니다.

이미 수집된 추가 정보(additional_info)를 문서 형태로 변환합니다.
웹 검색은 이미 완료되었으므로 추가 검색 없이 주어진 정보만 활용하세요.

중요한 규칙:
1. 원본 문서 설명(visual_description)의 정보와 완전히 독립적인 새로운 내용이어야 합니다.
2. 주어진 추가 정보(additional_info)를 문서 형태로 풍부하게 확장하세요.
3. multi-hop 질문의 정답(multi_hop_answer)이 이 문서에서 도출될 수 있어야 합니다.
4. 표, 차트, 그래프, 통계 등 시각적으로 볼 수 있는 정보들이 많을수록 좋습니다.
5. 새로운 문서는 A4 용지 한 페이지 분량을 넘어가지 않아야 합니다. 공백 포함 1,500자에서 2,000자 내외로 작성해야 합니다.
6. 문서는 한글로 작성하되, 필요한 경우 영어 등 다른 언어의 용어를 병기할 수 있습니다.
7. 반드시 문서만 출력하세요. 추가 설명이나 부가적인 텍스트는 포함하지 마세요."""

DOCUMENT_CONTENT_USER = """원본 문서: {visual_description}
Multi-hop 질문: {multi_hop_question}
Multi-hop 정답: {multi_hop_answer}
추가 정보: {additional_info}

위 정보를 바탕으로 가상의 문서 내용을 생성해주세요.
이 문서는 원본 문서와 독립적이면서, 추가 정보를 담고 있어야 하며,
multi-hop 질문의 정답이 이 문서에서 도출될 수 있어야 합니다."""

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
