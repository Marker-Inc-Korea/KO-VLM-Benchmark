## multi-choice question make

from google import genai
from google.genai import types


# numerical만 부분만 거의 다른 wrong answer
def generate_wrong_answer_numerical(
    image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
):

    # gemini 불러오기
    model = genai.Client(api_key=api_key)

    # find proper prompt
    instruction = (
        "당신은 주어진 [images, question, right answer]을 기반으로, 틀린 답변을 생성해내는 어시스턴트입니다.\n"
        "주어진 images와 question을 보고 wrong answer을 생성할 때, right answer가 가지고 있는 전체적인 문장 정보는 동일하게 유지하면서 **수치적인 정보**만 살짝 다르게 만들어야 합니다.\n"
        "이렇게 생성된 wrong answer는 right answer와 **수치적인 부분만 다르며**, AI나 사람이 맞추기 힘든 hard-case여야 합니다.\n"
        "생성된 wrong answer는 반드시 한국어로 출력해야하며, 주어진 정보들을 기반으로 명령을 잘 수행하세요."
    )

    query = f"{instruction}\nQuestion: {question}\nAnswer: {gt_answer}\n수치적인 부분만 달라진 wrong answer:"
    print(query)

    ## gemini inputs
    with open(image_path1, "rb") as f:
        image_bytes1 = f.read()
    with open(image_path2, "rb") as f:
        image_bytes2 = f.read()

    response = model.models.generate_content(
        model="gemini-2.5-pro",  # gemini-2.5-flash
        contents=[
            types.Part.from_bytes(
                data=image_bytes1,
                mime_type="image/jpeg",
            ),
            types.Part.from_bytes(
                data=image_bytes2,
                mime_type="image/jpeg",
            ),
            query,
        ],
    )

    output = response.text

    if output:
        # Clean unicode bugs if any
        cleaned_output = output.replace("\u0304", "")

        return cleaned_output

    else:
        return None


# 의미만 다른 wrong answer
def generate_wrong_answer_meaning(
    image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
):

    # gemini 불러오기
    model = genai.Client(api_key=api_key)

    # find proper prompt
    instruction = (
        "당신은 주어진 [images, question, right answer]을 기반으로, 틀린 답변을 생성해내는 어시스턴트입니다.\n"
        "주어진 images와 question을 보고 wrong answer을 생성할 때, right answer가 가지고 있는 수치적인 정보는 동일하게 유지하면서 **내용적으로 의미가 다르도록** 만들어야 합니다.\n"
        "이렇게 생성된 wrong answer는 right answer와 **내용적으로 상반되거나 거짓된 정보를 논리적으로 맞는 것처럼 설명하는 내용이 추가되어야 하며**, AI나 사람이 맞추기 힘든 hard-case여야 합니다.\n"
        "생성된 wrong answer는 반드시 한국어로 출력해야하며, 주어진 정보들을 기반으로 명령을 잘 수행하세요."
    )

    query = f"{instruction}\nQuestion: {question}\nAnswer: {gt_answer}\n내용적인 부분만 달라진 wrong answer:"
    print(query)

    ## gemini inputs
    with open(image_path1, "rb") as f:
        image_bytes1 = f.read()
    with open(image_path2, "rb") as f:
        image_bytes2 = f.read()

    response = model.models.generate_content(
        model="gemini-2.5-pro",  # gemini-2.5-flash
        contents=[
            types.Part.from_bytes(
                data=image_bytes1,
                mime_type="image/jpeg",
            ),
            types.Part.from_bytes(
                data=image_bytes2,
                mime_type="image/jpeg",
            ),
            query,
        ],
    )

    output = response.text

    if output:
        # Clean unicode bugs if any
        cleaned_output = output.replace("\u0304", "")

        return cleaned_output

    else:
        return None


# 풀이과정이 틀린 wrong answer
def generate_wrong_answer_explain(
    image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
):

    # gemini 불러오기
    model = genai.Client(api_key=api_key)

    # find proper prompt
    instruction = (
        "당신은 주어진 [images, question, right answer]을 기반으로, 틀린 답변을 생성해내는 어시스턴트입니다.\n"
        "주어진 images와 question을 보고 wrong answer을 생성할 때, right answer에 담긴 **전체적인 정보와 최종적인 정답은 동일**하지만, **풀이과정이 틀리도록** 만들어야합니다.\n"
        "이렇게 생성된 wrong answer는 right answer **보다 더 전문적으로 풀이를 해주고 최종적인 정답은 맞지만, 풀이과정에 약간의 오류가 있어야 하며**, AI나 사람이 맞추기 힘든 hard-case여야 합니다.\n"
        "생성된 wrong answer는 반드시 한국어로 출력해야하며, 주어진 정보들을 기반으로 명령을 잘 수행하세요."
    )

    query = f"{instruction}\nQuestion: {question}\nAnswer: {gt_answer}\n풀이과정이 약간 틀린 wrong answer:"
    print(query)

    ## gemini inputs
    with open(image_path1, "rb") as f:
        image_bytes1 = f.read()
    with open(image_path2, "rb") as f:
        image_bytes2 = f.read()

    response = model.models.generate_content(
        model="gemini-2.5-pro",  # gemini-2.5-flash
        contents=[
            types.Part.from_bytes(
                data=image_bytes1,
                mime_type="image/jpeg",
            ),
            types.Part.from_bytes(
                data=image_bytes2,
                mime_type="image/jpeg",
            ),
            query,
        ],
    )

    output = response.text

    if output:
        # Clean unicode bugs if any
        cleaned_output = output.replace("\u0304", "")

        return cleaned_output

    else:
        return None
