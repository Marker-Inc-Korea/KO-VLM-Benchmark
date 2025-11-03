# Introduction😋
![img](../그림3.png)  
AIhub에서 제공한, 시각화질의응답 데이터셋은 `문서 내 그림, 표, 그래프, 다이어그램(인포그래픽 포함) 등 시각화 자료`에 대한 이해 기반 질의응답 데이터로 시각 문서를 이해하고 문서의 내용에 관련된 질문에 대한 응답을 수행할 수 있는 데이터입니다.  
저희는 기존에 **한국어 기반 VLM**을 평가할 수 있는 데이터셋이 많지 않다는 것을 인지하고, 이를 위해 **🔥KO-VQA-Benchmark🔥** 데이터를 제작하게 되었습니다.  

저희 KO-VQA 데이터셋에서는 아래와 같은 VLM 모델의 성능을 평가할 수 있습니다.🔥🔥
```
- 한국어 기반 문서에 대한 이해
- 문서에 기반한 질문에 대한 VLM의 답변 능력
- 문서를 기반으로 질문에 대한 대답을 추론하는 능력
- 문서를 기반으로 질문에 대한 대답을 찾는 능력
- VLM 답변과 문서와의 alignment (숫자 표기 단위, 답변에 대한 표현 방법 등등)
```

저희가 제작한 KO-VQA 데이터셋은 기존 한국어 VLM 평가 데이터셋들과 비교하였을 때 아래와 같은 주요한 차별점이 있습니다!
```
실제 한국어 문서를 활용하여 데이터셋을 제작. (현실성🌟)
15개의 다양한 domain으로 구성된 문서를 활용. (다양성🌟)
```

# Environment
시각화자료질의응답 데이터셋을 기반으로 만든 한국어 VLM 벤치마크 데이터셋 **(KO-VQA)**

```
pytorch == 2.3.0 with cuda 12.1
transformers == 4.51.3
tokenizers == 0.21.1
qwen-vl-utils[decord] == 0.0.8
accelerate == 1.6.0
flash-attn == 2.7.4.post1
```

# Contents
1. [Introduction](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#introduction)😋
2. [How to make datasets](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#how-to-make-datasets)👽
3. [How to evaluate](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#how-to-evaluate)🦾
4. [Results](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#results)🌟
5. [References](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#references)

# How to make datasets👽
AIhub에서 제공한, [**시각화질의응답 데이터셋**](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71812)은 문서 내 그림, 표, 그래프, 다이어그램(인포그래픽 포함) 등 시각화 자료에 대한 이해 기반 질의응답 데이터로 시각 문서를 이해하고 문서의 내용에 관련된 질문에 대한 응답을 수행할 수 있는 데이터입니다.  
저희는 위 데이터셋을 벤치마크로 가공하기 위해, 아래의 데이터들을 활용하였습니다.😎
- 제공된 데이터에서 `Validation`만 활용
- 원천데이터에 제공된 `PDF image` 활용
- 라벨링데이터에 제공된 `각 domain별 json file` 활용
  
해당 데이터셋은 총 15개의 domain으로 구성되어있습니다.  
> 공공행정, 과학기술, 교육, 교통물류, 국토관리, 농축수산, 문화관광, 보건의료, 사회복지, 산업고용, 식품건강, 재난안전, 재정금융, 통일외교안보, 환경기상
   
저희는 각 domain 별로 random하게 100문항씩 추출하여, 총 1,500개의 VQA benchmark dataset을 구성했습니다.
아래는 간단한 instruction-answer에 대한 예시입니다!👇👇
```
# 공공행정
Instruction: <image> 2019년 밭농사의 기계화율은 1996년에 비해 얼마나 증가하나요?
Answer: 61.5조 원입니다.

# 농축수산
Instruction: <image> 2020년 공공기관 투자목표는 몇 조 원이니?
Answer: 밭농사의 기계화율은 21.9% 증가합니다.
```
  
KO-VQA 데이터셋의 일부 [subset](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark/blob/main/data/Sampled_%EC%8B%9C%EA%B0%81%ED%99%94_%EC%9E%90%EB%A3%8C_%EC%A7%88%EC%9D%98%EC%9D%91%EB%8B%B5_%EB%8D%B0%EC%9D%B4%ED%84%B0_benchmark_subset.csv)을 `/data/Sampled_시각화_자료_질의응답_데이터_benchmark_subset.csv`에서 확인하실 수 있습니다.🌞
> 전체 문항에 대해서는, 데이터 유출 및 데이터 저작권 문제로 인해 공유가 어렵습니다🤫

# How to evaluate🦾
시각화자료질의응답 Ko-VQA 데이터셋은 질문에 대한 답변을 문서에 기반하여 VLM이 얼마나 잘 답변하는지 알아보는 것에 중점을 둔 데이터셋입니다.  
따라서, 답변에서 제공된 `숫자 표기 단위`, `답변 표현 방법` 등등에 대해서 post-processing을 통해 VLM의 답변과 실제 answer를 비교하여 정확도를 측정합니다.  

---

저희는 `정규표현식`을 통해 VLM Output과 Answer에서 `숫자+단위`를 추출하고, 뽑혀진 두 개의 단어들이 정확히 일치한다면 정답이라고 평가합니다.  
또한 VLM이 내보내는 output이 `숫자+단위`를 형식을 지켜서 답변을 출력할 수 있도록, 아래의 prompt를 추가하여 평가를 진행합니다.
```
이미지를 보고 질문에 대한 답변을 제공해주세요. 이때, 반드시 이미지에 제공된 숫자와 단위를 명시해서 답변을 제공해야 합니다.

아래는 이미지에 제시된 숫자 단위가 '백만 원'일 때의 답변 예시입니다.
- 질문: 2017년도 국립청소년산림생태체험센터 건립사업에서 불용된 예산은 얼마인가요?
- 답변: 건립사업에서 불용된 예산은 총 7,131백만 원입니다.

아래는 이미지에 제시된 숫자 단위가 '천 명'일 때의 답변 예시입니다.
- 질문: 2008년 경제활동 인구는 몇 명인가요?
- 답변: 총 24,347천 명입니다.
```
  
아래는 정답 예시입니다!👇👇
```
Question: <image> 2020년 공공기관 투자목표는 몇 조 원이니?
VLM Output: 2020년 공공기관 투자목표는 61.5조 원입니다.
Answer: 61.5조 원입니다.
```
이때 각각 VLM output과 Answer에서는 `61.5조`가 추출되어 정답으로 평가됩니다!  
아래는 오답 예시입니다!👇👇
```
Question: <image> 2017년도 국가보훈처의 순국선열 애국지사사업기금 지출액은 얼마야?
Output: 2017년도 국가보훈처의 순국선열 애국지사사업기금 지출액은 19,305백만 원입니다.
Answer: 19,925백만 원이 순국선열 애국지사사업기금의 지출액입니다.
```
이때 각각 VLM output에서는 `[2017년, 19,305백]`이 추출되고, Answer에서는 `19,925백`이 추출되어 오답으로 간주됩니다.  

---

평가 코드는 아래 심플하게 돌려볼 수 있습니다!  
```bash
# Evaluation code
sh eval.sh
```
> You need to set `base_model` and `huggingfacce_token`.
  
# Results🌟
| Model | KO-VQA (Acc.) |
| ------------- | ------------- |
| `Gemini-2.5-pro` | **91.80** |
| `Gemini-2.5-flash` | 85.73 | 
| `Qwen2.5-VL-32B-Instruct` | **60.48** |
| `Qwen2.5-VL-7B-Instruct` | 53.27 |
| `Ovis2.5-2B (w/ thinking)` | 34.07 |
| `Ovis2.5-2B (w/o thinking)` | 31.27 |
| `VARCO-VISION-14B-HF` | 43.67 |
| `Gukbap-Ovis2-16B` | 34.80 |
| `Ovis2-16B` | 34.20 |
| `gemma-3-27b-it` | 34.20 |
| `Gukbap-Gemma3-27B-VL` | 33.60 |
| `Gukbap-Gemma3-12B-VL` | 30.13 |
| `Ovis2-34B` | 32.50 |
| `Gukbap-Ovis2-34B` | 31.93 |
| `gemma-3-12b-it` | 28.73 |
| `Bllossom-AICA-5B` | 20.67 |
   
# References
- [AIHub - 시각화질의응답 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71812)
