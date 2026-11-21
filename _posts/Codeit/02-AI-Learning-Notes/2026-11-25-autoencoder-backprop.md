---
title: "AutoEncoder - 02. 역전파와 기울기 계산"
date: 2026-11-25 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, backpropagation, gradient, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 학습 흐름과 기울기
- 순전파 → 손실(L) → 역전파 → 갱신을 반복
- 기울기(∂L/∂θ): 가중치를 늘릴 때 손실이 변하는 정도
- 역전파: 연쇄 법칙으로 뒤쪽 기울기부터 앞쪽으로 구함
- 출력 기울기의 계수 2/n: 제곱 미분에서 내려온 2, 평균에서 온 1/n

## 2. 역전파 공식 (2→1→2, 칸 수 n)

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 출력 기울기 | (2/n) × (복원값 − 입력) | $\frac{\partial L}{\partial \hat{x}_i}=\frac{2}{n}(\hat{x}_i-x_i)$ |
| 디코더 가중치 기울기 | 출력 기울기 × 잠재 표현 z | $\frac{\partial L}{\partial d_i}=\frac{\partial L}{\partial \hat{x}_i}z$ |
| 잠재 표현 기울기 | (출력 기울기 × 디코더 가중치 d)의 합 | $\frac{\partial L}{\partial z}=\sum_i\frac{\partial L}{\partial \hat{x}_i}d_i$ |
| 인코더 가중치 기울기 | 잠재 표현 기울기 × 입력 x | $\frac{\partial L}{\partial e_j}=\frac{\partial L}{\partial z}x_j$ |

📍[예시] $x=(2,3)$, $e=(0.5,0.2)$, $d=(1.0,0.5)$

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 잠재 표현 $z$ | $0.5\times2+0.2\times3$ | **1.6** |
| 복원값 $\hat{x}$ | $(1.0\times1.6,\ 0.5\times1.6)$ | **(1.6, 0.8)** |
| 손실 $L$ | $\frac{(-0.4)^2+(-2.2)^2}{2}$ | **2.5** |
| 출력 기울기 | $1\times(-0.4),\ 1\times(-2.2)$ | **(−0.4, −2.2)** |
| 디코더 기울기 | $(-0.4\times1.6,\ -2.2\times1.6)$ | **(−0.64, −3.52)** |
| 잠재 표현 기울기 | $-0.4\times1.0+(-2.2)\times0.5$ | **−1.5** |
| 인코더 기울기 | $(-1.5\times2,\ -1.5\times3)$ | **(−3, −4.5)** |

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
