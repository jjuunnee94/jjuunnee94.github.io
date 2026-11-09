---
title: "CNN - 08. ResNet"
date: 2026-11-12 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, cnn, resnet, residual, shortcut, projection, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. 깊게 쌓으면 생기는 문제

- 일반 CNN(Plain CNN)을 56층으로 쌓으면 20층보다 **훈련 오차부터** 높게 남음
- 깊이가 늘어도 최적화가 쉬워지는 것은 아님

| | 훈련 데이터 | 새 데이터 |
|:---:|:---:|:---:|
| 과적합 | 잘 맞힘 | 상대적으로 나쁨 |
| 56층 Plain CNN | 이미 나쁨 | 나쁨 |

- 훈련부터 나쁘므로 과적합이 아니라 **깊은 모델의 학습 어려움**
- 모든 깊은 망이 실패한다는 법칙은 아님

## 2. Residual: 입력을 보존하고 변화만 더한다

- shortcut : 입력 $x$를 그대로 전달하는 경로
- 주 경로 $F(x)$ (예: Conv → ReLU → Conv)와 shortcut을 **같은 위치끼리** 더한 뒤 ReLU

$$
H = F(x) + x, \qquad y = \text{ReLU}(H)
$$

- $F(x)=0$이면 $H=x$ → 아무것도 바꾸지 않는 동작을 쉽게 표현
- $x$, $F(x)$, $H$, $y$의 모양이 같아야 함

📍 **[예시]**

| | 왼쪽 위 | 오른쪽 위 | 왼쪽 아래 | 오른쪽 아래 |
|:---:|:---:|:---:|:---:|:---:|
| $x$ | 3 | −2 | 1 | 0 |
| $F(x)$ | −1 | 4 | −2 | 1 |
| $H = F(x)+x$ | $2$ | $2$ | $-1$ | $1$ |
| $y = \text{ReLU}(H)$ | 2 | 2 | **0** | 1 |

- shortcut이 있어도 $y$는 $x$와 같지 않음 (입력이 합산에 실제로 쓰이는지가 핵심)

🔍 **[비교]** 덧셈과 concat (두 입력 모두 `[1,16,14,14]`)

| 방식 | 하는 일 | 결과 모양 |
|:---:|:---:|:---:|
| 덧셈 (Residual) | 대응하는 값을 더함 | `[1,16,14,14]` |
| concat | 채널 묶음을 이어붙임 | `[1,32,14,14]` |

## 3. Identity와 Projection

- Identity shortcut : $x$를 바꾸지 않고 더함 (파라미터 없음)
- 조건 : 주 경로가 크기와 채널을 유지 ($k=3,\ s=1,\ p=1$이면 $\lfloor (14+2-3)/1 \rfloor + 1 = 14$)
- 모양이 바뀌면 그대로 더할 수 없음 (broadcast로도 맞지 않음)

| | 모양 |
|:---:|:---:|
| $F(x)$ | `[N,32,7,7]` |
| $x$ | `[N,16,14,14]` |

- Projection : shortcut에 **1×1 Conv (stride=2)** 를 두어 모양을 맞춤
- 1×1 Conv : 같은 위치의 채널 값들만 섞어서 새 채널을 만듦
- 공간 : $\lfloor (14 + 0 - 1)/2 \rfloor + 1 = 7$ → `[N,32,7,7]`
- Projection은 학습되는 변환 : $32\times(16\times1\times1+1)=544$

## 4. ResNet-34

- stage : 같은 출력 크기의 블록 묶음
- 입력 `3×224×224` → 7×7 Conv(stride 2) → MaxPool → `64×56×56`

| stage | Basic block 수 | 출력 모양 |
|:---:|:---:|:---:|
| 1 | 3 | `64×56×56` |
| 2 | 4 | `128×28×28` |
| 3 | 6 | `256×14×14` |
| 4 | 3 | `512×7×7` |

- Basic block : 3×3 Conv 두 개 + shortcut
- stage 경계(채널 증가, 공간 절반)의 shortcut은 projection
- 층 수 : $1 + 2\times(3+4+6+3) = 33$개 Conv + 마지막 FC 1개 = **34층**
- Global Average Pool : 채널마다 H×W를 평균 → `[N,512,7,7]` → `[N,512]`

| 방식 | 모양 | 마지막 FC 파라미터 |
|:---:|:---:|:---:|
| Flatten | `[N,25088]` | $25088\times1000$ 규모 |
| Global Average Pool | `[N,512]` | $512\times1000+1000=513{,}000$ |

- 원전 Basic block : `Conv → BN → ReLU → Conv → BN → (+x) → ReLU`
- 수업용 블록 : BN을 생략하고 합산 관계에 집중
- BN(Batch Normalization) : 중간값을 정규화하고 학습 가능한 크기·이동을 적용

## 5. 💻 [코드]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cnn_practice.ipynb)

```python
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)  # 크기 유지
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)  # 크기 유지

    def forward(self, x):
        out = F.relu(self.conv1(x))     # 주 경로 : Conv → ReLU
        out = self.conv2(out)           # 주 경로 : Conv (여기까지 F(x))
        return F.relu(out + x)          # shortcut x를 더한 뒤 ReLU

block = ResidualBlock(16)
x = torch.zeros(2, 16, 14, 14)                      # 이미지 2장, 16채널, 14x14
print(block(x).shape)                               # 입력과 같은 모양
print(sum(p.numel() for p in block.parameters()))   # Conv 2개 파라미터 수

# 모양이 바뀌는 경우 : 주 경로와 projection shortcut
main = nn.Conv2d(16, 32, 3, stride=2, padding=1)    # 채널 16→32, 14 → 7
short = nn.Conv2d(16, 32, 1, stride=2)              # 1x1 Conv, 14 → 7
print(main(x).shape)                                # 주 경로 모양
print(short(x).shape)                               # shortcut 모양 (같아야 더할 수 있음)
print(sum(p.numel() for p in short.parameters()))   # projection 파라미터 수
```

```
torch.Size([2, 16, 14, 14])
4640
torch.Size([2, 32, 7, 7])
torch.Size([2, 32, 7, 7])
544
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
