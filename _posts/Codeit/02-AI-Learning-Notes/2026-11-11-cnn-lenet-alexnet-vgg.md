---
title: "CNN - 07. LeNet-5, AlexNet, VGG-16"
date: 2026-11-11 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, cnn, lenet, alexnet, vgg, dropout, softmax, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. LeNet-5

- 흐름 : 입력 → (Conv → 축소) → (Conv → 축소) → 분류
- C = Convolution, S = Subsampling (인접 영역을 요약해 공간 축소)

| 단계 | 모양 `[C,H,W]` | 계산 |
|:---:|:---:|:---:|
| 입력 | `1×32×32` | |
| C1 (5×5, 필터 6개) | `6×28×28` | $\lfloor (32-5)/1 \rfloor + 1 = 28$ |
| S2 | `6×14×14` | $28 \to 14$ |
| C3 (5×5, 필터 16개) | `16×10×10` | $14 - 5 + 1 = 10$ |
| S4 | `16×5×5` | $10 \to 5$ |
| C5, F6, 출력 | 10개 | 클래스 점수 |

- C1 파라미터 : $6 \times (1\times5\times5 + 1) = 156$
- S층은 지금의 MaxPool과 세부 메커니즘이 다름

## 2. AlexNet

| | LeNet | AlexNet |
|:---:|:---:|:---:|
| 입력 | 한 채널 손글씨 | RGB 자연 이미지 |
| 클래스 수 | 10 | 1,000 |

- 구조 : Conv 5개(중간 Pool) → FC 3개 → 1,000개 점수
- 구조를 키우는 것만으로는 학습이 완성되지 않음 → 작용하는 대상이 다른 네 가지 선택을 함께 사용

| 선택 | 작용하는 곳 | 하는 일 |
|:---:|:---:|:---:|
| 데이터 증강 | 훈련 이미지 | 잘라내기, 좌우 반전으로 입력 상황을 다양화 |
| ReLU | Conv, FC의 중간값 | $\max(0, z)$로 비선형 변환 |
| Dropout | 훈련 중 일부 활성화 | 확률적으로 제외해서 과적합을 줄임 |
| GPU | 계산, 메모리 자원 | 큰 모델을 처리할 실행 자원 분담 |

- 과적합 : 훈련 데이터는 잘 맞히지만 새 데이터에서 나빠지는 현상
- Dropout : 훈련 때만 일부를 끄고, 평가 때는 전부 사용
- 네 가지를 하나의 연산이나 단일 성공 원인으로 묶지 않음

## 3. VGG-16

- 3×3 Conv를 반복하고, 블록 끝마다 Pool로 크기를 줄이는 규칙적인 구조
- 입력 : RGB `3×224×224`

| 블록 | 3×3 Conv 개수 | 채널 | Pool 뒤 크기 |
|:---:|:---:|:---:|:---:|
| 1 | 2 | 64 | 112×112 |
| 2 | 2 | 128 | 56×56 |
| 3 | 3 | 256 | 28×28 |
| 4 | 3 | 512 | 14×14 |
| 5 | 3 | 512 | 7×7 |

- 이후 : `512×7×7` → Flatten → FC 4096 → FC 4096 → FC 1000 → Softmax
- 가중치 층 : Conv 13개 + FC 3개 = **16개**, Conv마다 ReLU

### 한 블록

- 3×3 Conv (stride=1, padding=1) : 공간 크기 유지, 채널만 변경
- MaxPool 2×2 (stride=2) : 채널 유지, H·W만 절반
- 채널은 블록의 첫 Conv에서만 변경
- 깊이는 Conv 반복 횟수, 공간 크기는 Pool 경계에서 읽음
- 채널이 늘어나는 이유 : 앞쪽은 단순한 특징(선, 밝기), 뒤쪽은 그 조합이라 구분할 종류가 많아짐 (설계 관례)

### 수용영역과 가중치

- 3×3 Conv를 $n$번 쌓으면 $RF_n = 1 + 2n$ → 2개면 5×5, 3개면 7×7
- 같은 범위를 봐도 중간에 ReLU가 있어 5×5 한 개와 같은 함수는 아님
- 입력, 중간, 출력 채널을 모두 $C$로 맞추고 bias 제외

| | 계산 | $C=16$ |
|:---:|:---:|:---:|
| 3×3 두 개 | $2 \times (C\times C\times3\times3) = 18C^2$ | $4{,}608$ |
| 5×5 한 개 | $C\times C\times5\times5 = 25C^2$ | $6{,}400$ |

- 가중치 수 비교일 뿐, 정확도나 속도가 같은 비율로 좋아진다는 뜻은 아님

### FC와 Softmax

- FC = `nn.Linear` (파라미터 : $out \times (in + 1)$)
- 마지막 출력 개수는 클래스 수와 맞춤 (ImageNet 1000개)
- FC 사이에 ReLU 필요 (없으면 Linear 여러 개가 하나와 같아짐)

| 단계 | 모양 | 파라미터 |
|:---:|:---:|:---:|
| Flatten | `[25088]` ($512\times7\times7$) | 없음 |
| FC 4096 | `[4096]` | $4096\times(25088+1)=102{,}764{,}544$ |
| FC 4096 | `[4096]` | $4096\times(4096+1)=16{,}781{,}312$ |
| FC 1000 | `[1000]` | $1000\times(4096+1)=4{,}097{,}000$ |

- VGG-16 가중치 대부분은 Conv가 아니라 FC에 있음
- Softmax : 점수(logits)를 합이 1인 확률로 변환

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

📍 **[예시]** 점수 $[2, 1, 0]$

| 클래스 | 계산 | 확률 |
|:---:|:---:|:---:|
| 0 | $e^{2}/(e^{2}+e^{1}+e^{0}) = 7.389/11.107$ | 0.665 |
| 1 | $2.718/11.107$ | 0.245 |
| 2 | $1/11.107$ | 0.090 |

```python
import torch
from torch import nn
import torch.nn.functional as F

# LeNet 모양 확인
x = torch.zeros(1, 1, 32, 32)               # 이미지 1장, 1채널, 32x32
conv1 = nn.Conv2d(1, 6, 5)                  # 채널 1→6, 5x5 (padding=0)
conv2 = nn.Conv2d(6, 16, 5)                 # 채널 6→16, 5x5
pool = nn.MaxPool2d(2)                      # 2x2 풀링

x = pool(conv1(x)); print(x.shape)          # C1 → S2
x = pool(conv2(x)); print(x.shape)          # C3 → S4
print(sum(p.numel() for p in conv1.parameters()))   # C1 파라미터 수

# VGG 스타일 한 블록
conv_a = nn.Conv2d(64, 96, 3, padding=1)    # 채널 64→96, 크기 유지
conv_b = nn.Conv2d(96, 96, 3, padding=1)    # 채널 유지, 크기 유지
y = torch.zeros(1, 64, 40, 40)              # 블록 입력
y = F.relu(conv_a(y))                       # Conv → ReLU
y = F.relu(conv_b(y))                       # Conv → ReLU
y = pool(y)                                 # H, W 절반
print(y.shape)                              # 블록 출력 모양
print([sum(p.numel() for p in m.parameters()) for m in (conv_a, conv_b)])   # 층별 파라미터 수

# VGG의 첫 FC층
fc = nn.Linear(512 * 7 * 7, 4096)           # Flatten 뒤 25088 → 4096
print(sum(p.numel() for p in fc.parameters()))
```

```
torch.Size([1, 6, 14, 14])
torch.Size([1, 16, 5, 5])
156
torch.Size([1, 96, 20, 20])
[55392, 83040]
102764544
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
