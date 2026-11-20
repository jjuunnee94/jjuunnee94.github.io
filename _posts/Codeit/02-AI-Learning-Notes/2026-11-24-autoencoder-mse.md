---
title: "AutoEncoder - 01. 복원 구조와 작은 오토인코더"
date: 2026-11-24 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, autoencoder, reconstruction, mse, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
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
import numpy as np  # 배열 계산 라이브러리

def tiny_ae(x, e, d):  # 입력(x), 인코더 가중치(e), 디코더 가중치(d)
    z = (e * x).sum()  # 원소별 곱의 합 → 잠재 표현(z)
    reconstruction = d * z  # 디코더 가중치 × z → 복원값(x̂)
    mse = ((reconstruction - x) ** 2).mean()  # 차이 제곱의 평균 → 평균제곱오차
    return z, reconstruction, mse  # 세 값 반환

x = np.array([3.0, 1.0])  # 입력(x)
e = np.array([0.2, 0.6])  # 인코더 가중치(e)
d = np.array([1.5, 0.5])  # 디코더 가중치(d)
z, rec, mse = tiny_ae(x, e, d)  # 순전파 실행
print(z.round(2), rec.round(2), mse.round(2))  # 소수 둘째 자리까지 출력
```

```
1.2 [1.8 0.6] 0.8
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
