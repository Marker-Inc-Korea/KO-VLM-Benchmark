## multi-choice question make


from google import genai
from google.genai import types


# nemerical만 부분만 거의 다른 wrong answer
def generate_wrong_answer_numerical(
    image_path1, image_path2, image1_context, image2_context, question, gt_answer, api_key
):

    # gemini 불러오기
    model = genai.Client(api_key=api_key)

    # find proper prompt
    instruction = (
        "당신은 주어진 [images, question, right answer]을 기반으로, 틀린 답변을 생성해내는 어시스턴트입니다.\n"
        "주어진 images와 question을 보고 wrong answer을 생성할 때, right answer가 가지고 있는 전체적인 문장 정보는 동일하게 유지하면서 **수치적인 정보**만 살짝 다르게 만들어야 합니다. \n"
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


def generate_wrong_answer_meaning():
    pass


def generate_wrong_answer_order():
    pass
