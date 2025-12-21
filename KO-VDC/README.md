# Introduction: KO-VDC (visual context description choice)😋
![img2](../resources/그림2.png)
![img2_1](../resources/그림2_1.png)
AIhub에서 제공하는 멀티모달정보검색 데이터셋은 `다양한 표, 도식, 그래프 등등의 시각화 자료`를 포함한 한국어 문서에 대한 caption 정보를 가지고 있는 데이터셋입니다.
저희는 해당 데이터셋을 통해, **주어진 표/도식/그래프 만으로 문서의 설명문을 얼마나 잘 생성할 수 있는지**를 평가하기 위해 **🔥KO-VDC (Visual context Description Choice)🔥** 데이터셋을 제작하게 되었습니다!

저희 KO-VDC 데이터셋에서는 아래와 같은 VLM 모델의 성능을 평가할 수 있습니다.🔥🔥
```
- 복잡한 한국어 기반 표/도식/그래프 이해 능력
- 한국어 기반 표/도식/그래프에 대한 적절한 문서 설명문 생성 능력
- Long-Context 질문에 대한 모델의 답변 능력
```

저희가 제작한 KO-VDC 데이터셋은 기존 한국어 VLM 평가 데이터셋들과 비교하였을 때 아래와 같은 주요한 차별점이 있습니다!
```
실제 한국어 문서를 활용하여 데이터셋을 제작. (현실성🌟)
표/도식/그래프 기반 문서 생성 능력. (산업성🌟)
```

# Environment
멀티모달정보검색 데이터셋을 기반으로 만든 한국어 VLM 벤치마크 데이터셋 **(KO-VDC)**

```
pytorch == 2.3.0 with cuda 12.1
transformers == 4.57.1
qwen-vl-utils[decord] == 0.0.8
accelerate
flash-attn == 2.7.4.post1
```

# How to make datasets👽
멀티모달정보검색 데이터셋은 `다양한 표, 도식, 그래프 등등의 시각화 자료`를 포함한 한국어 문서를 제공하며, 보고서와 보도자료 2개의 카테고리로 구분되어 있습니다.
해당 데이터셋은 텍스트 문단에 대한 GT caption과 더불어서 위치에 대한 정보를 `bounding box`를 통해 알려줍니다.😋
저희는 각 보고서와 보도자료에서 100개씩 랜덤하게 추출하여 얻은 200개의 데이터를 KO-VDC 목적에 맞도록 가공하기 위해서, 총 **3차의 데이터 정제 과정**을 걸쳤습니다.🦾
- 1차: `Bounding box` 정보를 토대로, 텍스트 문단 정보 제거 (위의 대표사진 참고)
- 2차: `Gemini-2.5-pro`를 활용하여, GT caption를 더 풍부하게 가공 (`Gemini_GT`)
- 3차: `Gemini-2.5-pro`를 활용하여, 틀린 설명문 3가지를 추가적으로 제작

이렇게 만들어진 **KO-VDC** 데이터셋은, **VLM이 문서의 표/그래프/도식을 아무런 텍스트 정보 없이 얼마나 잘 이해할 수 있는지** 평가하게 됩니다!🔥
> 자세한 평가 방식은 아래의 section을 참고해주세요!

KO-VDC 데이터셋의 일부 [subset](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/blob/main/data/Gemini_sampled_%EB%A9%80%ED%8B%B0%EB%AA%A8%EB%8B%AC_%EC%A0%95%EB%B3%B4%EA%B2%80%EC%83%89_%EB%8D%B0%EC%9D%B4%ED%84%B0_benchmark_200_subset.xlsx)을 `(TODO)`에서 확인하실 수 있습니다.🌞
> 전체 문항에 대해서는, 데이터 유출 및 데이터 저작권 문제로 인해 공유가 어렵습니다🤫
> `Gemini_GT_1`이 정답 설명문이며, `Gemini_GT_2~4`는 틀린 설명문 입니다!

# How to evaluate🦾
KO-VDC 데이터셋은 **문서 내 표/도식/그래프만 참고하여 적절한 문서의 설명문을 생성하는 능력을 평가**하는 것이 주요한 목적입니다.
이때 설명문에 대한 명확한 정답은 존재하지 않고, `gemini-2.5-pro`로 만든 설명문을 ground truth로 사용하게 되면 편향이 발생하게 됩니다.🤫
그래서 저희는 앞서 언급한 것처럼, **틀린 설명문 3가지를 추가적으로 제작하여 가장 적절한 설명문을 고르는 객관식 문제**로 평가합니다.🔥

저희가 평가에 이용한 prompt는 다음과 같습니다:
```python
# choices는 string 변수
f'다음 주어진 [A, B, C, D] 보기 중 이미지에 있는 도식 정보를 가장 잘 설명하는 것을 고르시오. 답변은 무조건 알파벳이 먼저 나와야합니다.\n\n<보기>\n{choices}# 가장 적절한 설명문:'
```
여기서 `choices`는 각 설명문이 A/B/C/D에 하나씩 매핑되어 있는 변수인데, 저희는 평가를 진행하면서 일부 VLM들이 [Lost In the Middle](https://arxiv.org/abs/2307.03172) 현상이 발생하여 `A` 또는 `D`를 무지성 정답으로 선택하는 경향이 있다는 것을 확인했습니다.🤥
따라서 저희는 평가가 이루어질 때마다, 아래의 코드를 적용하여, **정답의 위치가 매번 바뀌도록 평가하여 일반화된 VLM의 성능을 평가**할 수 있었습니다!
```python
## shuffle
### choices 변수 생성
alpha_list = ['A', 'B', 'C', 'D']
index_shuffle = random.sample([1,2,3,4], 4)
alpha_num = 0
choices = ''
gt_alpha = ''
for index in index_shuffle:
    if index == 1:
        choices += f'{alpha_list[alpha_num]}. {Gemini_GT_1}\n\n'
        gt_alpha = alpha_list[alpha_num]

    elif index == 2:
        choices += f'{alpha_list[alpha_num]}. {Gemini_GT_2}\n\n'

    elif index == 3:
        choices += f'{alpha_list[alpha_num]}. {Gemini_GT_3}\n\n'

    elif index == 4:
        choices += f'{alpha_list[alpha_num]}. {Gemini_GT_4}\n\n'

    alpha_num += 1
```

---

평가 코드는 아래 심플하게 돌려볼 수 있습니다!
```bash
# Evaluation code
sh eval_VDC.sh
```
> You need to set `base_model` and `huggingfacce_token`.

# Results🌟
| Model | KO-VDC (Acc.) |
| ------------- | ------------- |
| `Closed-model` | ---- |
| `Gemini-3-pro` | `Not Yet` | 71.60 |
| `Gemini-2.5-pro` | **97.50** |
| `Gemini-2.5-flash` | 85.50 |
| `Open-model` | ---- |
| `Qwen3-VL-30B-A3B-Instruct` | `OOM` |
| `Qwen3-VL-8B-Instruct` | **68.50** |
| `Qwen3-VL-4B-Instruct` | 42.50 |
| `Qwen2.5-VL-32B-Instruct` | `OOM` |
| `Qwen2.5-VL-7B-Instruct` | 39.50 |
| `Ovis2.5-9B` | **52.50** |
| `Ovis2.5-2B` | 32.25 |
| `Ovis2-34B` | 22.50 |
| `Ovis2-16B` | 26.00 |
| `Gemma-3-27b-it` | 38.00 |
| `Gemma-3-12b-it` | 30.25 |
| `Gukbap-Ovis2-16B` | 23.50 |
| `VARCO-VISION-2.0-14B-HF` | 36.00 |
| `VARCO-VISION-14B-HF` | 4.00 |
| `Bllossom-AICA-5B` | 2.00 |

# References
- [AIHub - 멀티모달정보검색 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71813)
