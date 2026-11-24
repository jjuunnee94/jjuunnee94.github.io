---
title: "AutoEncoder - 01. 복원 구조와 작은 오토인코더"
date: 2026-11-24 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, autoencoder, reconstruction, mse, codeit-sprint, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. 오토인코더(AutoEncoder) 구조
- 인코더(E): 입력(x) → 잠재 표현(z) 
- 디코더(D): 잠재 표현(z) → 복원값(x̂), 원본(x)은 보지 못함
- 정답(타깃) = 입력 자기 자신, 별도 라벨 없음
- 분류는 정답 1개(라벨), 오토인코더는 정답이 입력과 같은 개수

## 2. 작은 오토인코더(2→1→2)와 평균제곱오차
- 📐[공식] 잠재 표현·복원값: $z = e_1 x_1 + e_2 x_2,\quad \hat{x} = (d_1 z,\ d_2 z)$
- 📐[공식] 평균제곱오차(MSE): $L = \frac{1}{n}\sum_{i=1}^{n}(\hat{x}_i - x_i)^2$
- 순서: 차이 → 제곱 → 평균, 제곱은 한 번만
- 제곱은 부호 상쇄 방지, 완벽 복원이면 $L=0$

📍[예시] $x=(3,1)$, $e=(0.2,0.6)$, $d=(1.5,0.5)$

| 잠재 표현(z) | 복원값(x̂) | 차이(x̂−x) | 제곱 | 평균제곱오차(L) |
|:---:|:---:|:---:|:---:|:---:|
| $0.2\times3+0.6\times1$ = **1.2** | $(1.5\times1.2,\ 0.5\times1.2)$ = **(1.8, 0.6)** | $(-1.2,\ -0.4)$ | $(1.44,\ 0.16)$ | $\frac{1.44+0.16}{2}$ = **0.8** |

## 3. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
import torch  # 파이토치
import torch.nn as nn  # 신경망 층 모음

encoder = nn.Linear(2, 1, bias=False)  # 인코더: 2칸 → 1칸, 편향 없음
decoder = nn.Linear(1, 2, bias=False)  # 디코더: 1칸 → 2칸, 편향 없음
encoder.weight = nn.Parameter(torch.tensor([[0.2, 0.6]]))  # 인코더 가중치(e) 직접 지정, 모양 (1, 2)
decoder.weight = nn.Parameter(torch.tensor([[1.5], [0.5]]))  # 디코더 가중치(d) 직접 지정, 모양 (2, 1)
x = torch.tensor([[3.0, 1.0]])  # 입력(x), 모양 (1, 2)

with torch.no_grad():  # 기울기 기록 없이 계산
    z = encoder(x)  # 잠재 표현(z)
    rec = decoder(z)  # 복원값(x̂)
    mse = nn.MSELoss()(rec, x)  # 평균제곱오차(MSE)
print(z, rec, mse)  # 결과 출력
```

```
tensor([[1.2000]]) tensor([[1.8000, 0.6000]]) tensor(0.8000)
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
