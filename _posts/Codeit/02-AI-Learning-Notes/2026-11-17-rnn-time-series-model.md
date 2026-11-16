---
title: "RNN - 03. RNN으로 시계열 예측"
date: 2026-11-17 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, rnn, 시계열예측, linear, 입력창, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. Linear 출력층 (head)

- RNN의 마지막 상태 $(B,\ H)$는 $H$개의 특징, 예측은 숫자 하나
- `nn.Linear(H, 1)`: $H$개 특징을 가중합해 예측 $(B,\ 1)$로 변환
- weight $(1,\ H)$, 입력 $(B,\ H)$는 `x @ W.T + b`로 계산

📐[공식]

$$\hat y = w_0h_0 + w_1h_1 + \cdots + w_{H-1}h_{H-1} + b$$

- $\hat y$: 예측값, $h_i$: 마지막 상태의 $i$번째 칸, $w_i$: 그 칸의 가중치, $b$: 편향

📍[예시] $H=3,\ h=[0.7,\ -0.3,\ 0.5],\ w=[1.0,\ -2.0,\ 0.5],\ b=0.2$

| 상태 칸 | $h_i$ | $w_i$ | $w_ih_i$ |
|:---:|:---:|:---:|:---:|
| 0 | 0.7 | 1.0 | 0.7 |
| 1 | −0.3 | −2.0 | 0.6 |
| 2 | 0.5 | 0.5 | 0.25 |
| 합 + $b$ | | | 0.7 + 0.6 + 0.25 + 0.2 = **1.75** |

## 2. 입력창과 정답

- 입력창: 앞 $L$일의 값, 정답: 바로 다음 날의 값
- 창을 하루씩 이동하며 (입력창, 정답) 쌍을 여러 개 생성
- 정답 날도 데이터 안에 있어야 하므로 창의 개수 = $N - L$ ($N$: 전체 일수)
- $X$: (창 수, $L$, $F$) = $(B,\ T,\ F)$, $y$: (창 수, 1)
- 창 하나가 예제 하나이고, 창의 길이 $L$이 시점 수 $T$
- 실제 데이터는 값을 줄여서(예: 100으로 나눠) 입력

📍[예시] 10일치 `[5, 7, 6, 9, 8, 11, 10, 13, 12, 15]`, $L=4$

| 창 번호 $i$ | 입력창 (날 $i$ ~ $i+3$) | 정답 (날 $i+4$) |
|:---:|:---:|:---:|
| 0 | [5, 7, 6, 9] | **8** |
| 1 | [7, 6, 9, 8] | **11** |
| 2 | [6, 9, 8, 11] | **10** |
| 3 | [9, 8, 11, 10] | **13** |
| 4 | [8, 11, 10, 13] | **12** |
| 5 | [11, 10, 13, 12] | **15** |

- 창 개수 $10-4=6$, $X$는 $(6,\ 4,\ 1)$, $y$는 $(6,\ 1)$

## 3. 입력창 배치 계산

- 같은 RNN 가중치를 모든 창이 공유
- 각 창(행)의 상태는 독립적으로 계산, 창마다 $h_0=0$에서 시작
- 창끼리 값이 겹쳐도 한 창의 마지막 상태를 다음 창으로 넘기지 않음
- 창을 하나씩 돌린 결과와 배치로 한 번에 돌린 결과는 같음

📍[예시] $H=1,\ W_{ih}=0.6,\ W_{hh}=0.5$, 편향 없음, 창 `[2, 0]`과 `[0, 2]`

| 창 | 시점 | $x_t$ | $W_{ih}x_t$ | $W_{hh}h_{t-1}$ | $a_t$ | $h_t=\tanh(a_t)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [2, 0] | 1 | 2 | 1.2 | 0 | 1.2 | 0.833655 |
| [2, 0] | 2 | 0 | 0 | 0.416827 | 0.416827 | **0.394254** |
| [0, 2] | 1 | 0 | 0 | 0 | 0 | 0 |
| [0, 2] | 2 | 2 | 1.2 | 0 | 1.2 | **0.833655** |

- 값은 같고 순서가 달라서 마지막 상태가 다름

## 4. 예측 모델의 shape 흐름

- `output`에서 마지막 시점을 고르고(`output[:, -1, :]`), Linear에 전달
- $B$는 끝까지 유지, $T$는 `last`에서 사라지고, $F$는 RNN을 지나며 $H$로 바뀜
- 학습 전 예측은 무작위 가중치의 출력

| 단계 | 모양 | 예 ($B=8,\ T=6,\ F=1,\ H=5$) |
|:---:|:---:|:---:|
| 입력 $X$ | $(B,\ T,\ F)$ | $(8,\ 6,\ 1)$ |
| `output` | $(B,\ T,\ H)$ | $(8,\ 6,\ 5)$ |
| `last` | $(B,\ H)$ | **$(8,\ 5)$** |
| 예측 | $(B,\ 1)$ | $(8,\ 1)$ |

## 5. 💻 [코드]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/rnn_practice.ipynb)

```python
import torch                                                         # 파이토치
from torch import nn                                                 # 신경망 모듈

data = torch.tensor([5., 7., 6., 9., 8., 11., 10., 13., 12., 15.])   # 10일치, 모양 (10,)
L = 4                                                                # 입력창 길이
X = torch.stack([data[i:i+L] for i in range(len(data) - L)])         # 창 6개를 쌓음, (6, 4)
X = X.unsqueeze(-1)                                                  # 끝에 크기 1인 축 추가, (6, 4, 1)
y = data[L:].unsqueeze(-1)                                           # 정답, (6, 1)

class SimpleRNN(nn.Module):                                          # 모델 클래스
    def __init__(self, hidden_size):                                 # 층을 만드는 곳
        super().__init__()                                           # 부모 클래스 준비
        self.rnn = nn.RNN(1, hidden_size, batch_first=True)          # 입력 1칸, 상태 H칸
        self.fc = nn.Linear(hidden_size, 1)                          # 상태 H칸 → 예측 1칸

    def forward(self, x):                                            # 계산 순서를 정하는 곳
        output, h_n = self.rnn(x)                                    # 모든 시점의 상태 (B, T, H)
        last = output[:, -1, :]                                      # 마지막 시점의 상태 (B, H)
        return self.fc(last)                                         # 예측 (B, 1)

model = SimpleRNN(8)                                                 # 상태 8칸인 모델
pred = model(X)                                                      # forward 실행, 예측 (6, 1)
print(X.shape, y.shape, pred.shape)                                  # 세 모양 확인
```

```
torch.Size([6, 4, 1]) torch.Size([6, 1]) torch.Size([6, 1])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
