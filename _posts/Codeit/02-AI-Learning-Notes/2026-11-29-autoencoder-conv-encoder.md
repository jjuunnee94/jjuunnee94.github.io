---
title: "AutoEncoder - 06. 합성곱 인코더"
date: 2026-11-29 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, cnn, conv2d, feature-map, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 합성곱 인코더(Conv2d)
- 작은 창(커널)을 이미지 위에서 옮기며, 겹친 칸끼리 곱해 모두 더하고 편향을 더해 값 하나를 만듦
- 값들이 모여 특징맵 한 장, 커널 수 = 출력 채널 수
- 계산한 값에 ReLU를 적용해 음수는 0으로 바꿈
- 스트라이드(s): 커널이 한 번에 움직이는 칸 수, 2이면 크기가 줄어듦
- 패딩(p): 가장자리에 덧붙이는 0의 칸 수
- 모양은 [채널, 높이, 너비], 파이토치 입력은 (샘플 수, 채널, 높이, 너비) 네 축
- 강의 구조: 입력 [1, 28, 28] → 잠재 표현 [4, 14, 14]

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 출력 크기 | 내림((입력 크기 + 2×패딩 − 커널 크기) ÷ 스트라이드) + 1 | $H_{out}=\left\lfloor\frac{H+2p-k}{s}\right\rfloor+1$ |
| 파라미터 수 | 입력 채널 × 출력 채널 × 커널² + 출력 채널 | $C_{in}C_{out}k^2+C_{out}$ |

📍[예시] 출력 크기: 입력 8×8, $k=3$, $s=2$, $p=1$

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 분자 | 8 + 2×1 − 3 | 7 |
| 나누기 | 7 ÷ 2 | 3.5 |
| 내림 | ⌊3.5⌋ | **3** |
| +1 | 3 + 1 | **4** (출력 4×4) |

📍[예시] 값 하나 계산: 커널 K = [[1,0,−1],[0,2,0],[1,0,0]], 편향 $b=-3$

| 위치 | 3×3 구역 P | 겹쳐 곱해 합 | 편향 더하기 | ReLU 후 |
|:---:|:---:|:---:|:---:|:---:|
| A | [[1,2,0],[0,1,3],[2,0,1]] | 1+0+0 + 0+2+0 + 2+0+0 = **5** | 5−3 = 2 | **2** |
| B | [[0,1,0],[1,0,0],[0,0,1]] | **0** | 0−3 = −3 | **0** |

- 파라미터 수: $1\times4\times3^2+4=$ **40개** (강의 구조)
- 강의 구조는 원소 수가 입력 1×28×28 = 784, 잠재 표현 4×14×14 = 784로 같아, 칸 수를 줄이는 병목과는 성격이 다름
- 학습 결과에서 채널 3이 모두 0이었지만, 이것만으로 각 채널의 역할을 단정하지 않음

## 2. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
import torch  # 파이토치
import torch.nn as nn  # 신경망 층 모음

encoder = nn.Sequential(  # 층을 순서대로 연결
    nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),  # 입력 채널 1, 출력 채널 4, 커널 3×3, 두 칸씩 이동, 가장자리 1칸 덧댐
    nn.ReLU()  # 음수는 0
)
x = torch.zeros(1, 1, 28, 28)  # 모양 확인용 입력: (샘플 1, 채널 1, 28×28)
z = encoder(x)  # 인코더 통과
print(x.shape, z.shape)  # 입력 모양, 잠재 표현 모양 출력
```

```
torch.Size([1, 1, 28, 28]) torch.Size([1, 4, 14, 14])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
