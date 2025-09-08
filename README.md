# Introduction😋


각각의 VLM 데이터셋에 대한 자세한 리뷰 및 코드 설명  
1️⃣[KO-VQA🔱](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/tree/main/KO-VQA)  
2️⃣[KO-MMQ🔱](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/tree/main/KO-MMQ)
  
# Contents
1. [Introduction😋](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#introduction)😋
2. [Contributions👽](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#how-to-evaluate)🦾
3. [Results🌟](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#results)🌟
4. [References](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#references)
5. [Acknowledgement](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#acknowledgement)
  
# Contributions👽
## KO-VQA
-
<자세한 KO-VQA에 대한 설명과 예제 사진은 [KO-VQA README🔱](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/tree/main/KO-VQA) 참고해주세요!>  
-
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
  
## KO-MMQ
  
# Results🌟
| Model | KO-VQA (Acc.) | KO-MMQ (Acc.) |
| ------------- | ------------- | ------------- |
| `Gemini-2.5-pro` | **91.80** | NaN |
| `Gemini-2.5-flash` | 85.73 | NaN |
| `Qwen2.5-VL-32B-Instruct` | **60.48** | NaN |
| `Qwen2.5-VL-7B-Instruct` | 53.27 | 39.50 |
| `Ovis2.5-9B (w/ thinking)` | NaN | NaN |
| `Ovis2.5-2B (w/ thinking)` | NaN | 32.25 |
| `VARCO-VISION-14B-HF` | 43.67 | 4.00 |
| `Gukbap-Ovis2-16B` | 34.80 | NaN |
| `Ovis2-16B` | 34.20 | NaN |
| `gemma-3-27b-it` | 34.20 | NaN |
| `Gukbap-Gemma3-27B-VL` | 33.60 | NaN |
| `Gukbap-Gemma3-12B-VL` | NaN | 30.25 |
| `Ovis2-34B` | 32.50 | NaN |
| `Gukbap-Ovis2-34B` | 31.93 | NaN |
| `gemma-3-12b-it` | 28.73 | 30.25 |
| `Bllossom-AICA-5B` | 20.67 | 2.00 |
   
# References
- [AIHub - 시각화질의응답 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71812)
- [AIHub - 멀티모달정보검색 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71813)

# Acknowledgement 
This research was supported by the Korea Institute for Advancement of Technology (KIAT) grant funded by the Korean Government (MOTIE) (RS-2024-00416131, HRD Program for Industrial Innovation)
