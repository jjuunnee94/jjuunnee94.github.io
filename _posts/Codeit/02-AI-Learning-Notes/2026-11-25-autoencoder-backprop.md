---
title: "AutoEncoder - 02. 역전파와 파라미터 갱신"
date: 2026-11-25 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, autoencoder, backpropagation, gradient-descent, learning-rate, codeit-sprint, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. 학습 흐름과 기울기
- 순전파 → 손실(L) → 역전파 → 갱신을 반복
- 기울기(∂L/∂θ): 가중치를 늘릴 때 손실이 변하는 정도, 음수면 손실 감소
- 학습률(η): 이동량 크기 조절, 크면 지나칠 수 있음
- 역전파: 연쇄 법칙으로 뒤쪽 기울기부터 구함

## 2. 공식 (2→1→2, 칸 수 n)

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 출력 기울기 | (2/n) × (x̂−x) | $\frac{\partial L}{\partial \hat{x}_i}=\frac{2}{n}(\hat{x}_i-x_i)$ |
| 디코더 가중치 기울기 | 출력 기울기 × z | $\frac{\partial L}{\partial d_i}=\frac{\partial L}{\partial \hat{x}_i}z$ |
| 잠재 표현 기울기 | (출력 기울기 × d)의 합 | $\frac{\partial L}{\partial z}=\sum_i\frac{\partial L}{\partial \hat{x}_i}d_i$ |
| 인코더 가중치 기울기 | z의 기울기 × x | $\frac{\partial L}{\partial e_j}=\frac{\partial L}{\partial z}x_j$ |
| 가중치 갱신 | 현재 − 학습률 × 기울기 | $\theta_{new}=\theta-\eta\frac{\partial L}{\partial\theta}$ |

📍[예시] $x=(2,3)$, $e=(0.5,0.2)$, $d=(1.0,0.5)$, $\eta=0.1$

| 가중치 | 기울기 | 이동량(η×기울기) | 갱신 값 |
|:---:|:---:|:---:|:---:|
| $e_1$ | $-1.5\times2$ = −3 | −0.3 | 0.5−(−0.3) = **0.8** |
| $e_2$ | $-1.5\times3$ = −4.5 | −0.45 | 0.2−(−0.45) = **0.65** |
| $d_1$ | $-0.4\times1.6$ = −0.64 | −0.064 | 1.0−(−0.064) = **1.064** |
| $d_2$ | $-2.2\times1.6$ = −3.52 | −0.352 | 0.5−(−0.352) = **0.852** |

- $z=1.6$, 출력 기울기 (−0.4, −2.2), $\partial L/\partial z=-1.5$, 손실 2.5 → 약 1.58

## 3. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
x = np.array([2.0, 3.0])  # 입력(x)
e = np.array([0.5, 0.2])  # 인코더 가중치(e)
d = np.array([1.0, 0.5])  # 디코더 가중치(d)
eta = 0.1  # 학습률(η)
n = x.size  # 입력 칸 수(n)
z = (e * x).sum()  # 순전파: 잠재 표현(z)
rec = d * z  # 복원값(x̂)
grad_rec = (2 / n) * (rec - x)  # 출력 기울기
grad_d = grad_rec * z  # 디코더 가중치 기울기
grad_z = (grad_rec * d).sum()  # 잠재 표현 기울기
grad_e = grad_z * x  # 인코더 가중치 기울기
e = e - eta * grad_e  # 인코더 가중치 갱신
d = d - eta * grad_d  # 디코더 가중치 갱신
print(e.round(3), d.round(3))  # 갱신된 가중치 출력
```

```
[0.8  0.65] [1.064 0.852]
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
