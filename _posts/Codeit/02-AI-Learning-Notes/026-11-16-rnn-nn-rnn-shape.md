---
title: "RNN - 02. nn.RNN의 입력과 반환값"
date: 2026-11-16 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, rnn, nn-rnn, shape, 은닉상태, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. B · F · H와 RNNCell의 shape

- $B$: 한 번에 계산하는 예제 수
- $F$: 한 시점의 입력 칸 수
- $H$: 상태 $h$의 칸 수 (은닉 특징 수)
- Cell의 입력에는 시간 축이 없음, 시퀀스는 시점마다 호출
- 대괄호 겹 수가 축의 수, $(1,\ 2)$는 `[[0.5, -1.0]]`처럼 두 겹

| 객체 | 기호 | 예 ($B=1,\ F=1,\ H=2$) |
|:---:|:---:|:---:|
| 현재 입력 $x_t$ | $(B,\ F)$ | $(1,\ 1)$ |
| 직전 상태 $h_{t-1}$ | $(B,\ H)$ | $(1,\ 2)$ |
| 새 상태 $h_t$ | $(B,\ H)$ | $(1,\ 2)$ |

## 2. H칸 상태 계산

📐[공식]

$$a_t = W_{ih}x_t + W_{hh}h_{t-1} + b, \qquad h_t = \tanh(a_t)$$

- $W_{ih}$: $(H,\ F)$, $W_{hh}$: $(H,\ H)$, $b$: $(H,)$ (두 편향의 합)
- 새 상태의 $i$번째 칸은 $W$의 $i$번째 행으로 계산
- 공식은 값을 세로 벡터로 본 표기, 코드는 예제가 한 행이라 $x_tW_{ih}^{T} + h_{t-1}W_{hh}^{T} + b$로 계산 (같은 숫자가 가로로 나옴)

📍[예시] $x_t=[[3]],\ h_{t-1}=[[0.5,\ -1.0]],\ W_{ih}=\begin{bmatrix}0.4\\-0.2\end{bmatrix},\ W_{hh}=\begin{bmatrix}0.3&0.1\\-0.5&0.2\end{bmatrix},\ b=[0.1,\ -0.1]$

| 상태 칸 | $W_{ih}x_t$ | $W_{hh}h_{t-1}$ | $+\,b$ | $a_t$ | $h_t=\tanh(a_t)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.4×3 = 1.2 | 0.3×0.5 + 0.1×(−1.0) = 0.05 | 0.1 | 1.35 | **0.8741** |
| 1 | −0.2×3 = −0.6 | −0.5×0.5 + 0.2×(−1.0) = −0.45 | −0.1 | −1.15 | **−0.8178** |

⚠️[주의] $H=2$는 시점이 2번이나 층이 2개라는 뜻이 아니다.

## 3. nn.RNN의 입력과 반환값

- RNNCell: 한 시점 계산, 시점 반복은 직접 for 문
- `nn.RNN`: 전체 시퀀스를 받아 시점 반복을 안에서 처리
- `batch_first=True`: 입력과 `output`의 첫 축을 $B$로 둠, `h_n`의 축 순서는 그대로
- `output`: 마지막 층의 **모든 시점** 상태
- `h_n`: 층·방향별 **마지막 시점** 상태, 시간 축이 없음
- $K$: 층 수, $D$: 방향 수, 지금은 $K=D=1$
- 초기 상태를 넣지 않으면 0으로 시작, 호출 사이에 자동으로 이어지지 않음

📍[예시] $B=1,\ T=3,\ F=1,\ H=2$

| 객체 | 기호 | 예 |
|:---:|:---:|:---:|
| 입력 $x$ | $(B,\ T,\ F)$ | $(1,\ 3,\ 1)$ |
| `output` | $(B,\ T,\ H)$ | $(1,\ 3,\ 2)$ |
| `h_n` | $(K{\times}D,\ B,\ H)$ | $(1,\ 1,\ 2)$ |

## 4. 마지막 시점 상태 고르기

- `output[:, -1, :]`: `:`은 그 축 유지, `-1`은 마지막 위치 하나
- 숫자로 고른 축은 사라지므로 $(B,\ T,\ H)$에서 $T$가 빠져 $(B,\ H)$
- `h_n[-1]`: 마지막 층의 마지막 상태 $(B,\ H)$, 한 층·한 방향이면 `output[:, -1, :]`와 같은 값

📍[예시] `output`이 $(2,\ 4,\ 2)$일 때

| 예제 | `output` (시점 0 ~ 3) | `output[:, -1, :]`의 줄 |
|:---:|:---:|:---:|
| 0 | [1, 2], [3, 4], [5, 6], [7, 8] | **[7, 8]** |
| 1 | [0, 1], [0, 2], [0, 3], [0, 4] | **[0, 4]** |

- 결과는 `[[7, 8], [0, 4]]`, 모양 $(2,\ 2)$

## 5. H가 바뀌면 달라지는 shape

- $B,\ T,\ F$는 그대로이고 상태의 **마지막 축**만 $H$로 바뀜
- 두 모델의 가중치는 따로 무작위로 초기화됨

📍[예시] $B=2,\ T=3,\ F=1$

| 객체 | $H=2$ | $H=4$ |
|:---:|:---:|:---:|
| 입력 | $(2,\ 3,\ 1)$ | $(2,\ 3,\ 1)$ |
| `output` | $(2,\ 3,\ \mathbf{2})$ | $(2,\ 3,\ \mathbf{4})$ |
| `h_n` | $(1,\ 2,\ \mathbf{2})$ | $(1,\ 2,\ \mathbf{4})$ |

## 6. nn.RNN의 파라미터 수

📐[공식]

$$H\times F + H\times H + 2H$$

- $W_{ih}$: $H\times F$, $W_{hh}$: $H\times H$, 편향 두 개: $H$개씩
- $T$와 $B$는 식에 없음, 같은 가중치를 모든 시점·예제에 재사용
- $H$가 커지면 $W_{hh}$가 $H^2$으로 늘어남

📍[예시] `nn.RNN(2, 3)` ($F=2,\ H=3$, 시점 수 10)

| $W_{ih}$ ($H\times F$) | $W_{hh}$ ($H\times H$) | 편향 ($2H$) | 합계 |
|:---:|:---:|:---:|:---:|
| 3×2 = 6 | 3×3 = 9 | 2×3 = 6 | **21** |

## 7. 💻 [코드]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/rnn_practice.ipynb)

```python
import torch                                              # 파이토치
from torch import nn                                      # 신경망 모듈

torch.manual_seed(0)                                      # 무작위 초기값 고정
cell = nn.RNNCell(1, 2)                                   # 입력 1칸, 상태 2칸
seq = torch.tensor([[2.0], [0.0], [-1.0]])                # 시점 3개 × 입력 1칸, 모양 (3, 1)
h = torch.zeros(1, 2)                                     # h0 = 0, 모양 (B, H)
for xt in seq:                                            # 시점 하나씩 꺼냄
    h = cell(xt.reshape(1, 1), h)                         # (현재 입력, 직전 상태) → 새 상태
    print(h.shape)                                        # 상태의 모양

rnn = nn.RNN(1, 2, batch_first=True)                      # 입력 1칸, 상태 2칸, 입력 모양 (B, T, F)
x3 = seq.reshape(1, 3, 1)                                 # (3, 1) → (1, 3, 1)
output, h_n = rnn(x3)                                     # 모든 시점을 한 번에 계산
print(output.shape, h_n.shape)                            # 모든 시점 상태, 마지막 상태

last = output[:, -1, :]                                   # 모든 예제, 마지막 시점, 모든 상태 칸
print(last.shape)                                         # 시간 축이 사라진 모양

x_b2 = torch.tensor([2., 0., -1., 1., 3., 0.]).reshape(2, 3, 1)  # 예제 2 × 시점 3 × 입력 1
rnn_h4 = nn.RNN(1, 4, batch_first=True)                   # 상태 4칸
out4, hn4 = rnn_h4(x_b2)                                  # H=4 결과
print(out4.shape, hn4.shape)                              # 마지막 축이 4
```

```
torch.Size([1, 2])
torch.Size([1, 2])
torch.Size([1, 2])
torch.Size([1, 3, 2]) torch.Size([1, 1, 2])
torch.Size([1, 2])
torch.Size([2, 3, 4]) torch.Size([1, 2, 4])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
