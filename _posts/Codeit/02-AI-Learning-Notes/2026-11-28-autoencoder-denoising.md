---
title: "AutoEncoder - 05. 노이즈 제거 오토인코더"
date: 2026-11-28 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, denoising, noise, clamp, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 노이즈 제거 오토인코더(DAE)
- 입력은 노이즈 낀 이미지(noisy), 비교 정답은 같은 표본의 깨끗한 이미지(clean)
- 기본 AE는 clean → 복원 → clean, DAE는 noisy → 복원 → clean
- 정답을 noisy로 두면 노이즈까지 따라 만들 수 있어 clean으로 유지
- 자르기(clip): 픽셀은 0~1 범위라서 0보다 작으면 0, 1보다 크면 1로 바꿈

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 노이즈 입력 | 자르기(깨끗한 입력 + 0.25 × 표준정규 노이즈, 0, 1) | $\tilde{x}=\text{clip}(x+0.25\,\varepsilon,\ 0,\ 1)$ |
| 손실 | 평균제곱오차(복원값, 깨끗한 입력) | $L=\text{MSE}\big(D(E(\tilde{x})),\ x\big)$ |

📍[예시] $x=(0.2,0.9,1.0,0.0)$, $\varepsilon=(0.8,-1.2,2.0,-0.4)$

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 노이즈 크기 | $0.25 \times \varepsilon$ | (0.2, −0.3, 0.5, −0.1) |
| 더하기 | $x + $ 노이즈 | (0.4, 0.6, **1.5**, **−0.1**) |
| 자르기 | 0~1로 | **noisy = (0.4, 0.6, 1.0, 0.0)** |
| 가정한 복원값 | 모델 출력이라고 가정 | $\hat{x}$ = (0.3, 0.7, 0.9, 0.1) |
| 복원값 오차 | clean과 비교: $\frac{0.01+0.04+0.01+0.01}{4}$ | **0.0175** |
| 기준선 | noisy와 clean 비교: $\frac{0.04+0.09+0+0}{4}$ | **0.0325** |

- 복원값이 개선됐는지 보려면 두 오차를 모두 정답(clean) 기준으로 구해 비교
  1. noisy ↔ clean 오차 = 기준선
  2. 복원값 ↔ clean 오차 = 모델 점수
  3. 모델 점수 < 기준선이면 노이즈가 줄어든 것

| 픽셀 | clean | noisy | noisy 차이 | 복원값 | 복원값 차이 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.2 | 0.4 | 0.2 | 0.3 | **0.1** (가까워짐) |
| 2 | 0.9 | 0.6 | 0.3 | 0.7 | **0.2** (가까워짐) |
| 3 | 1.0 | 1.0 | 0 | 0.9 | 0.1 (멀어짐) |
| 4 | 0.0 | 0.0 | 0 | 0.1 | 0.1 (멀어짐) |
| 평균제곱오차 | | | $\frac{0.13}{4}$ = **0.0325** | | $\frac{0.07}{4}$ = **0.0175** |

## 2. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
import torch  # 파이토치
import torch.nn as nn  # 신경망 층 모음

clean = torch.tensor([[0.2, 0.9, 1.0, 0.0]])  # 깨끗한 픽셀(clean)
eps = torch.tensor([[0.8, -1.2, 2.0, -0.4]])  # 표준정규 노이즈 (실제는 torch.randn(clean.shape)로 생성)
noisy = (clean + 0.25 * eps).clamp(0, 1)  # 0.25배 노이즈를 더하고 0~1로 자름
rec = torch.tensor([[0.3, 0.7, 0.9, 0.1]])  # 복원값(가정, 실제는 model(noisy))
mse = nn.MSELoss()  # 평균제곱오차 계산기
print(noisy)  # 노이즈 입력 출력
print(mse(rec, clean), mse(noisy, clean))  # 복원값 오차, 기준선 오차
```

```
tensor([[0.4000, 0.6000, 1.0000, 0.0000]])
tensor(0.0175) tensor(0.0325)
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
