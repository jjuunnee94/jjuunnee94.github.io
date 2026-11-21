---
title: "RNN - 07. LSTM의 여러 상태 칸과 nn.LSTM"
date: 2026-11-21 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, rnn, lstm, nn-lstm, 은닉상태, 셀상태, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. 상태가 여러 칸일 때의 갱신

- 은닉 상태 $h$와 셀 상태 $c$가 각각 $H$칸, 같은 번호의 칸끼리 계산
- 칸 번호는 시간 순서가 아니고, $H=2$는 시점이 2번이라는 뜻이 아님
- 칸마다 자기 망각 게이트 $f$를 가지므로 직접 경로의 배율도 칸마다 자기 $f$들의 곱, 예: 같은 칸의 $f$가 세 번 연속 0.9이면 $0.9^3=0.729$

📍[예시] $B=1,\ H=2$, 게이트 값이 주어진 경우

| 칸 | $c_{t-1}$ | $f$ | $i$ | $g$ | $o$ | 보존분 $f\odot c_{t-1}$ | 반영분 $i\odot g$ | $c_t$ | $h_t=o\odot\tanh(c_t)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.6 | 0.5 | 0.5 | −0.5 | 0.7 | 0.3 | −0.25 | **0.05** | **0.034971** |
| 1 | −0.4 | 0.3 | 0.6 | 0.5 | 0.9 | −0.12 | 0.3 | **0.18** | **0.160273** |

- 기호: 이전 셀 상태 $c_{t-1}$, 망각 게이트 $f$, 입력 게이트 $i$, 후보 $g$, 출력 게이트 $o$
- 결과 $c_t=[0.05,\ 0.18]$, $h_t=[0.034971,\ 0.160273]$, 모양은 둘 다 $(1,\ 2)$

## 2. 행렬곱과 원소곱

- 게이트 생성: 입력과 이전 은닉 상태의 여러 칸을 섞는 **행렬곱**
- 상태 갱신: 같은 칸끼리 곱하는 **원소곱** $\odot$

📐[공식]

$$\text{묶음 가중합} = x_tW_{ih}^{T} + b_{ih} + h_{t-1}W_{hh}^{T} + b_{hh}$$

- $(B,\ 4H)$를 마지막 축에서 4등분하면 $i,\ f,\ g,\ o$ 가중합 각 $(B,\ H)$

| 객체 | 모양 |
|:---:|:---:|
| 입력 $x_t$ / $W_{ih}$ | $(B,\ F)$ / $(4H,\ F)$ |
| 이전 은닉 상태 $h_{t-1}$ / $W_{hh}$ | $(B,\ H)$ / $(4H,\ H)$ |
| 묶음 가중합 | $(B,\ 4H)$ |
| 경로별 가중합 ($i,\ f,\ g,\ o$) | 각 $(B,\ H)$ |

## 3. nn.LSTM의 입력과 반환값

- `nn.LSTMCell`: 한 시점의 $(h,\ c)$ 반환, `nn.LSTM`: 시간 축 반복을 한 번에 처리
- 반환: `output, (h_n, c_n)`, `nn.RNN`의 `output, h_n`에 셀 상태가 추가됨
- `output`: 모든 시점의 은닉 상태만 모임, `h_n`, `c_n`: 마지막 시점 상태, 시간 축 없음
- 같은 호출에서 `output[:, -1, :]`와 `h_n[0]`은 같은 값, `c_n`은 저장 상태라 `h_n`과 다른 값
- 초기 상태 $(h_0,\ c_0)$는 각각 $(1,\ B,\ H)$, 생략하면 0에서 시작
- 은닉 상태: 밖으로 드러낼 특징, 셀 상태: 보존과 갱신이 쌓이는 저장 공간

📍[예시] $B=1,\ T=3,\ F=1,\ H=2$, 한 층·한 방향, `batch_first=True`

| 객체 | 기호 | 모양 |
|:---:|:---:|:---:|
| 입력 $x$ | $(B,\ T,\ F)$ | $(1,\ 3,\ 1)$ |
| `output` | $(B,\ T,\ H)$ | $(1,\ 3,\ 2)$ |
| `h_n` | $(K{\times}D,\ B,\ H)$ | $(1,\ 1,\ 2)$ |
| `c_n` | $(K{\times}D,\ B,\ H)$ | $(1,\ 1,\ 2)$ |

## 4. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/rnn_practice.ipynb)

```python
import torch                                       # 파이토치
from torch import nn                               # 신경망 모듈

torch.manual_seed(0)                               # 무작위 초기값 고정
lstm = nn.LSTM(1, 2, batch_first=True)             # 입력 1칸, 상태 2칸, 입력 모양 (B, T, F)
x = torch.tensor([[[0.5], [-0.3], [0.8]]])         # 예제 1개 × 시점 3개 × 입력 1칸, (1, 3, 1)
output, (h_n, c_n) = lstm(x)                       # 모든 시점을 한 번에 계산
print(output.shape, h_n.shape, c_n.shape)          # 세 반환값의 모양

last = output[:, -1, :]                            # 마지막 시점의 은닉 상태
print(last.shape)                                  # 시간 축이 사라진 모양
```

```
torch.Size([1, 3, 2]) torch.Size([1, 1, 2]) torch.Size([1, 1, 2])
torch.Size([1, 2])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
