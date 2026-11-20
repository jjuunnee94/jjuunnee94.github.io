---
title: "AutoEncoder 기호 및 공식 정리 2"
date: 2026-11-20 17:30:00 +0900
categories: [Codeit, '04-Study-Note']
tags: tags: [codeit, autoencoder, backpropagation, gradient-descent, bottleneck, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

#### 🔍[기호] 역전파·병목 기호 · 이름 · 뜻

| 기호 | 이름 | 뜻 |
|:---:|:---:|:---:|
| **$\theta$** | **파라미터** | 학습으로 바뀌는 값 ($e_1,e_2,d_1,d_2$) |
| **$\eta$** | **학습률** | 한 번에 이동하는 크기 |
| **$\dfrac{\partial L}{\partial \theta}$** | **기울기** | $\theta$를 늘릴 때 $L$이 변하는 정도 |
| **$m$** | **입력 차원** | 입력의 칸 수 |
| **$k$** | **잠재 차원** | 잠재 표현($z$)의 칸 수 |

#### 📍[공식] 역전파 (2→1→2, $n=2$)

- **출력 기울기** (평균제곱오차(MSE) 미분)
$$ \frac{\partial L}{\partial \hat{x}_i} = \frac{2}{n}(\hat{x}_i - x_i) $$

- **디코더 가중치 기울기** ($\hat{x}_i = d_i z$)
$$ \frac{\partial L}{\partial d_i} = \frac{\partial L}{\partial \hat{x}_i}\, z $$

- **잠재 표현 기울기** (두 출력의 합)
$$ \frac{\partial L}{\partial z} = \sum_{i}\frac{\partial L}{\partial \hat{x}_i}\, d_i $$

- **인코더 가중치 기울기** ($z = \sum_j e_j x_j$)
$$ \frac{\partial L}{\partial e_j} = \frac{\partial L}{\partial z}\, x_j $$

#### 📍[공식] 파라미터 갱신

- **경사 하강법** (기울기 반대 방향으로 이동)
$$ \theta_{\text{new}} = \theta - \eta\,\frac{\partial L}{\partial \theta} $$

#### 🔍[비교] 병목

- **병목 조건**: $k < m$ (정보 손실 발생)
- **압축 비율**: $m / k$

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
