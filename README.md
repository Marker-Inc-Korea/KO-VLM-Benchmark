# Introduction😋
오늘날 해외에서 멀티모달에 대한 관심이 커짐에 따라 foundation model 및 benchmark dataset이 다양하게 제작되고 공유되고 있습니다.  
하지만, 해외 멀티모달 benchmark 경우 질문/답변에 대한 구성이 `영어`로 이루어져 있어, 한국어 능력에 대한 평가를 정확하게 할 수 없습니다.😵   
이에 따라, 저희는 기존에 한국어 기반 VLM을 평가할 수 있는 데이터셋이 많지 않다는 것을 인지하였고,   
오픈소스 기여와 발전을 위해 **🔥KO-VLM Benchmark dataset🔱**를 제작하게 되었습니다.🤗  
  
**AI-Hub**와 **공공데이터포털**에서는 한국어 기반의 Vision Question Answering (VQA) 데이터셋을 풍부하게 제공하고 있습니다.🌎    
저희는 AI-Hub에서 제공하는 2가지 데이터셋과 공동데이터포털에서 제공하는 여러가지 문서들을 수집 및 활용하여 **🔱KO-VLM Benchmark dataset** 제작하였습니다.    
이를 활용해 국내/외 있는 Vision-Language Model (VLM)들의 한국어 문서 및 질문 이해 능력을 측정할 수 있습니다.😎   

**KO-VLM Benchamrk dataset🔱**은 총 3가지 데이터셋으로 구성되어 있습니다.
```
- KO-VQA🔱: `다양한 도메인의 한국어 문서 이해 능력` 및 `문서 기반의 답변 추론 능력`에 대해 평가
- KO-VDC🔱: `한국어 시각화 도식 자료 이해 능력` 및 `도식 기반의 설명문 생성/이해 능력`에 대해 평가
- KO-OCRAG🔱: `한국어 문서 OCR 능력` 및 `문서에 등장하는 Visual Context parsing 능력`에 대해 평가
```
  
위의 3가지 데이터셋은 기존 한국어 VLM Benchmark 데이터셋과는 확연한 **차별점**이 있습니다.
```
실제 한국어 문서를 활용하여 데이터셋을 제작. (현실성🌟)
문서와 도식을 기반으로 정답을 찾아야하는 문제들로 구성. (추론형🌟)
```

---
각각의 VLM 데이터셋에 대한 자세한 리뷰 및 코드 설명은 아래를 참고해주세요😋  
1️⃣[KO-VQA🔱](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/tree/main/KO-VQA)  
2️⃣[KO-VDC🔱](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/tree/main/KO-VDC)  
3️⃣[KO-OCRAG🔱](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/tree/main/KO-OCRAG)

---
  
# Contents
1. [Introduction😋](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#introduction)
2. [Contributions👽](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#how-to-evaluate)
3. [Results🌟](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#results)
4. [References](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#references)
5. [Acknowledgement](https://github.com/Marker-Inc-Korea/KO-VQA-Benchmark?tab=readme-ov-file#acknowledgement)
  
# Contributions👽
## 1️⃣KO-VQA
저희 KO-VQA 데이터셋에서는 아래와 같은 VLM 모델의 성능을 평가할 수 있습니다.🔥🔥  
```
- 한국어 기반 문서에 대한 이해
- 문서에 기반한 질문에 대한 VLM의 답변 능력
- 문서를 기반으로 질문에 대한 대답을 추론하는 능력
- 문서를 기반으로 질문에 대한 대답을 찾는 능력
- VLM 답변과 문서와의 alignment (숫자 표기 단위, 답변에 대한 표현 방법 등등)
```
> 자세한 KO-VQA에 대한 설명과 예제들은 [KO-VQA README🔱](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/tree/main/KO-VQA) 참고해주세요!
  
저희가 제작한 KO-VQA 데이터셋은 기존 한국어 VLM 평가 데이터셋들과 비교하였을 때 아래와 같은 주요한 차별점이 있습니다!
```
실제 한국어 문서를 활용하여 데이터셋을 제작. (현실성🌟)
15개의 다양한 domain으로 구성된 문서를 활용. (다양성🌟)
```
  
## 2️⃣KO-VDC
(TODO)

## 3️⃣KO-OCRAG
(TODO)
  
# Results🌟
| Model | KO-VQA (Acc.) | KO-VDC (Acc.) | KO-OCRAG (Avg.) |
| ------------- | ------------- | ------------- | ------------- |
| `Gemini-2.5-pro` | **91.80** | **97.50** | NaN |
| `Gemini-2.5-flash` | 85.73 | 85.50 | NaN |
| `Qwen2.5-VL-32B-Instruct` | **60.48** | NaN | NaN |
| `Qwen2.5-VL-7B-Instruct` | 53.27 | 39.50 | NaN | 
| `Ovis2.5-9B (w/ thinking)` | NaN | NaN | NaN |
| `Ovis2.5-2B (w/ thinking)` | 34.07 | 32.25 | NaN |
| `VARCO-VISION-2.0-14B` | NaN | NaN | NaN |
| `VARCO-VISION-14B-HF` | 43.67 | 4.00 | NaN |
| `Gukbap-Ovis2-16B` | 34.80 | NaN | NaN |
| `Ovis2-16B` | 34.20 | NaN | NaN |
| `gemma-3-27b-it` | 34.20 | NaN | NaN |
| `Gukbap-Gemma3-27B-VL` | 33.60 | NaN | NaN |
| `Gukbap-Gemma3-12B-VL` | 30.13 | 30.25 | NaN |
| `Ovis2-34B` | 32.50 | NaN | NaN |
| `Gukbap-Ovis2-34B` | 31.93 | NaN | NaN |
| `gemma-3-12b-it` | 28.73 | 30.25 | NaN |
| `Bllossom-AICA-5B` | 20.67 | 2.00 | NaN |
> KO-OCRAG: `(WER+CER+rough2)/3`
   
# References
- [AIHub - 시각화질의응답 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71812)
- [AIHub - 멀티모달정보검색 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71813)
- [공공데이터포털](https://www.data.go.kr/bbs/rcr/selectRecsroom.do?pageIndex=1&originId=PDS_0000000000000703)

# Acknowledgement 
This research was supported by the Korea Institute for Advancement of Technology (KIAT) grant funded by the Korean Government (MOTIE) (RS-2024-00416131, HRD Program for Industrial Innovation)

# TODO
- [ ] KO-VDC 설명추가
- [ ] KO-OCRAG 코드추가
- [ ] 전체적인 Code Update
- [ ] Models 벤치마크 완료하기
