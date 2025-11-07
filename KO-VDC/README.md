# Introduction: KO-VDC (visual context description choice)😋
![img2](../그림2.png)   
![img2_1](../그림2_1.png)   
AIhub에서 제공하는 멀티모달정보검색 데이터셋은 `다양한 표, 도식, 그래프 등등의 시각화 자료`를 포함한 한국어 문서에 대한 caption 정보를 가지고 있는 데이터셋입니다.  
저희는 해당 데이터셋을 통해, **주어진 표/도식/그래프 만으로 문서의 설명문을 얼마나 잘 생성할 수 있는지**를 평가하기 위해 **🔥KO-VDC (Visual context Description Choice)🔥** 데이터셋을 제작하게 되었습니다!

저희 KO-VDC 데이터셋에서는 아래와 같은 VLM 모델의 성능을 평가할 수 있습니다.🔥🔥
```
- 복잡한 한국어 기반 표/도식/그래프 이해 능력
- 한국어 기반 표/도식/그래프에 대한 문서 생성 능력
- 문서에 존재하는 visual information에 대한 text description 생성 능력
```

저희가 제작한 KO-VDC 데이터셋은 기존 한국어 VLM 평가 데이터셋들과 비교하였을 때 아래와 같은 주요한 차별점이 있습니다!
```
실제 한국어 문서를 활용하여 데이터셋을 제작. (현실성 🌟)
표/도식/그래프 기반 문서 생성 능력 (산업성 🌟)
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
(TODO)
  
KO-VDC 데이터셋의 일부 [subset](https://github.com/Marker-Inc-Korea/KO-VLM-Benchmark/blob/main/data/Gemini_sampled_%EB%A9%80%ED%8B%B0%EB%AA%A8%EB%8B%AC_%EC%A0%95%EB%B3%B4%EA%B2%80%EC%83%89_%EB%8D%B0%EC%9D%B4%ED%84%B0_benchmark_200_subset.xlsx)을 `(TODO)`에서 확인하실 수 있습니다.🌞
> 전체 문항에 대해서는, 데이터 유출 및 데이터 저작권 문제로 인해 공유가 어렵습니다🤫

# How to evaluate🦾
(TODO)

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
- [AIHub - 멀티모달정보검색 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71813)
