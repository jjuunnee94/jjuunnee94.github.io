---
title: "AutoEncoder - 07. 합성곱 디코더"
date: 2026-11-30 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, cnn, conv-transpose, sigmoid, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 합성곱 디코더(전치합성곱, ConvTranspose2d)
- 입력값 하나에 커널을 곱해 출력의 여러 위치에 놓고, 겹치는 칸은 더해 더 큰 이미지를 만듦
- 다음 입력값은 스트라이드(s)만큼 옆으로 옮겨 놓음, 커널이 스트라이드보다 크면 이웃 기여가 겹침
- 패딩(p): 합성곱과 반대로 만든 출력의 바깥 테두리를 p칸씩 뺌
- 입력 채널이 여러 개면 모든 채널의 기여를 합치고 편향을 더한 뒤 Sigmoid로 0~1 복원값을 만듦
- 커널의 값은 학습으로 정해짐

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 출력 크기 | (입력 크기 − 1) × 스트라이드 − 2 × 패딩 + 커널 크기 | $H_{out}=(H-1)\,s-2p+k$ |
| 파라미터 수 | 입력 채널 × 출력 채널 × 커널² + 출력 채널 | $C_{in}C_{out}k^2+C_{out}$ |
| 시그모이드 | 1 ÷ (1 + e^(−x)), 결과는 0~1 | $\sigma(x)=\frac{1}{1+e^{-x}}$ |

- 강의 구조: (14−1)×2 − 2×1 + 4 = **28**, 파라미터 4×1×4²+1 = **65개** (인코더 40 + 디코더 65 = 105개)
- 시그모이드: 큰 값은 1에, 작은 값은 0에 가깝게 만들고 σ(0) = 0.5 (슬라이드에 없음)

📍[예시] 1차원: 입력 (1, 2, 3), 커널 (1, 0, 2), $s=2$, $p=0$

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 출력 길이 | (3−1)×2 − 0 + 3 | **7** |
| 입력 1 놓기 (위치 0~2) | 1 × (1, 0, 2) | (1, 0, 2) |
| 입력 2 놓기 (위치 2~4) | 2 × (1, 0, 2) | (2, 0, 4) |
| 입력 3 놓기 (위치 4~6) | 3 × (1, 0, 2) | (3, 0, 6) |
| 겹친 칸 더하기 | 위치 2: 2+2, 위치 4: 4+3 | **(1, 0, 4, 0, 7, 0, 6)** |

패딩 1을 주면 양 끝 1칸씩을 빼서 출력 길이 (3−1)×2 − 2×1 + 3 = **5**, 값은 **(0, 4, 0, 7, 0)**

📍[예시] 편향 $b=-2$ 더하고 Sigmoid 적용

| 위치 | 합 | 편향 더하기 | Sigmoid |
|:---:|:---:|:---:|:---:|
| 0 | 0 | −2 | **0.119** |
| 1 | 4 | 2 | **0.881** |
| 2 | 0 | −2 | **0.119** |
| 3 | 7 | 5 | **0.993** |
| 4 | 0 | −2 | **0.119** |

## 2. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
import torch  # 파이토치
import torch.nn as nn  # 신경망 층 모음

decoder = nn.Sequential(  # 층을 순서대로 연결
    nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),  # 입력 채널 4, 출력 채널 1, 커널 4×4, 두 칸 간격, 테두리 1칸 빼기
    nn.Sigmoid()  # 0~1 값으로 변환
)
z = torch.zeros(1, 4, 14, 14)  # 모양 확인용 잠재 표현: (샘플 1, 채널 4, 14×14)
rec = decoder(z)  # 디코더 통과
print(z.shape, rec.shape)  # 잠재 표현, 복원값 모양 출력
```

```
torch.Size([1, 4, 14, 14]) torch.Size([1, 1, 28, 28])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
