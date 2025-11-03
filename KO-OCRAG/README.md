# Introduction😋
![img](../그림3.png)   
![img](../그림4.png)   
`공공데이터셋포털`에서는 실제 산업에서 활용되는 다양하고 복잡한 구조를 가진 한국어 문서를 풍부하게 제공하고 있습니다.  
저희는 공공데이터를 직접 수집하고 가공하여, 기존의 단순한 **한국어 문서 OCR**이 아닌 **RAG용 한국어 문서 OCR**을 평가할 수 있는 데이터셋인 **🔥KO-OCRAG🔥**를 제작하게 되었습니다.  

저희 KO-OCRAG 데이터셋에서는 아래와 같은 VLM 모델의 성능을 평가할 수 있습니다.🔥🔥
```
- 복잡한 한국어 문서 구조 이해
- 고해상도 한국어 문서 OCR 능력
- 문서에 존재하는 visual information에 대한 text description 생성 능력
- RAG parsing에 적합한 description 생성 능력
```

저희가 제작한 KO-OCRAG 데이터셋은 기존 한국어 VLM 평가 데이터셋들과 비교하였을 때 아래와 같은 주요한 차별점이 있습니다!
```
고해상도의 복잡한 구조를 가진 다양한 한국어 문서. (다양성 🌟)
RAG parsing에 적합한 description 생성 능력. (RAG 🌟)
```

# Environment
`공공데이터셋포털`에서 수집한 데이터셋을 기반으로 만든 한국어 VLM 벤치마크 데이터셋 **(KO-OCRAG)**

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
`공공데이터셋포털`에서 제공하는 다양한 데이터셋은, 실제 산업에서 활용되는 다양하고 복잡한 구조를 가진 한국어 문서로 구성되어 있습니다.
(TODO)
  
KO-OCRAG 데이터셋의 일부 [subset]()을 `???`에서 확인하실 수 있습니다.🌞
> 전체 문항에 대해서는, 데이터 유출 및 데이터 저작권 문제로 인해 공유가 어렵습니다🤫

# How to evaluate🦾
KO-OCRAG 데이터셋은 (TODO)

---

평가 코드는 아래 심플하게 돌려볼 수 있습니다!  
```bash
(TODO)
```
> You need to set `base_model` and `huggingfacce_token`.
  
# Results🌟
| Model | KO-OCRAG (Acc.) |
| ------------- | ------------- |
| `Gemini-2.5-pro` | NaN |
| `Gemini-2.5-flash` | NaN | 
| `Qwen2.5-VL-32B-Instruct` | NaN |
| `Qwen2.5-VL-7B-Instruct` | NaN |
| `Ovis2.5-2B (w/ thinking)` | NaN |
| `Ovis2.5-2B (w/o thinking)` | NaN |
| `VARCO-VISION-14B-HF` | NaN |
| `Gukbap-Ovis2-16B` | NaN |
| `Ovis2-16B` | NaN |
| `gemma-3-27b-it` | NaN |
| `Gukbap-Gemma3-27B-VL` | NaN |
| `Gukbap-Gemma3-12B-VL` | NaN |
| `Ovis2-34B` | NaN |
| `Gukbap-Ovis2-34B` | NaN |
| `gemma-3-12b-it` | NaN |
| `Bllossom-AICA-5B` | NaN |
   
# References
- [공공데이터셋포털](https://www.data.go.kr/index.do)
