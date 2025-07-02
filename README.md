# KO-VQA-Benchmark
시각화자료질의응답 데이터셋을 기반으로 만든 VLM 벤치마크 데이터셋

# Contents
1. Introduction😋
2. How to make datasets👽
3. How to evaluation🦾
4. Results🌟
5. References

# Introduction😋
AIhub에서 제공한, 시각화질의응답 데이터셋은 `문서 내 그림, 표, 그래프, 다이어그램(인포그래픽 포함) 등 시각화 자료`에 대한 이해 기반 질의응답 데이터로 시각 문서를 이해하고 문서의 내용에 관련된 질문에 대한 응답을 수행할 수 있는 데이터입니다.  
저희는 기존에 **한국어 기반 VLM**을 평가할 수 있는 적절한 데이터셋이 없다는 것을 인지하고, 이를 위해 **🔥KO-VQA-Benchmark🔥** 데이터를 제작하게 되었습니다.  

저희 KO-VQA 데이터셋에서는 아래와 같은 VLM 모델의 성능을 평가할 수 있습니다.🔥🔥
```
- 한국어 기반 문서에 대한 이해
- 문서에 기반한 질문에 대한 VLM의 답변 능력
- 문서를 기반으로 질문에 대한 대답을 추론하는 능력
- 문서를 기반으로 질문에 대한 대답을 찾는 능력
- VLM 답변과 문서와의 alignment (숫자 표기 단위, 답변에 대한 표현 방법 등등)
```
  
# How to make datasets👽
AIhub에서 제공한, **시각화질의응답 데이터셋**은 문서 내 그림, 표, 그래프, 다이어그램(인포그래픽 포함) 등 시각화 자료에 대한 이해 기반 질의응답 데이터로 시각 문서를 이해하고 문서의 내용에 관련된 질문에 대한 응답을 수행할 수 있는 데이터입니다.  
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
  
KO-VQA 데이터셋의 일부 subset을 `/data/Sampled_시각화_자료_질의응답_데이터_benchmark_subset.csv`에서 확인하실 수 있습니다.🌞
> 전체 문항에 대해서는, 데이터 유출 및 데이터 저작권 문제로 인해 공유가 어렵습니다🤫

# How to evaluation🦾
(답변 전처리 방법 설명)

(코드 실행 방법 설명)
```
# 코드 환경
```

```bash
# Evaluation code

```
(option 설명)

# Results🌟
(TO-DO)

# References
- (AIHub - 시각화질의응답 데이터셋][https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71812]  
