---
title: "RNN - 06. LSTM 수치 계산과 LSTMCell"
date: 2026-11-20 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, rnn, lstm, lstmcell, 게이트, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. LSTM 한 시점의 계산

- 순서: ① 네 경로의 가중합 → ② 활성화로 게이트 생성 → ③ 셀 상태 갱신 → ④ 은닉 상태 생성 → ⑤ 다음 시점으로 전달
- 네 경로: 입력 게이트 $i$, 망각 게이트 $f$, 후보 $g$, 출력 게이트 $o$
- 경로마다 입력 가중치 $W_x$, 상태 가중치 $W_h$, 편향 $b$를 따로 가짐

📐[공식]

$$a_q = W_x x_t + W_h h_{t-1} + b, \qquad c_t = f \odot c_{t-1} + i \odot g, \qquad h_t = o \odot \tanh(c_t)$$

- $a_q$: 경로 $q$의 가중합, $i,\ f,\ o$는 $\sigma(a_q)$, $g$는 $\tanh(a_g)$
- $c_t$: 셀 상태, $h_t$: 은닉 상태, $\odot$: 같은 칸끼리 곱

📍[예시] $x_t=0.5,\ h_{t-1}=-0.2,\ c_{t-1}=0.4$

| 경로 | $W_x$ | $W_h$ | $b$ | $W_x x_t$ | $W_h h_{t-1}$ | $a_q$ | 게이트 값 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 입력 게이트 $i$ | 0.7 | −0.4 | 0 | 0.35 | 0.08 | 0.43 | $\sigma$ → **0.605874** |
| 망각 게이트 $f$ | 0.9 | 0.3 | 0.1 | 0.45 | −0.06 | 0.49 | $\sigma$ → **0.620106** |
| 후보 $g$ | −0.6 | 0.5 | 0.2 | −0.3 | −0.1 | −0.2 | $\tanh$ → **−0.197375** |
| 출력 게이트 $o$ | 0.4 | 0.8 | −0.1 | 0.2 | −0.16 | −0.06 | $\sigma$ → **0.485004** |

| 단계 | 식 | 계산 | 값 |
|:---:|:---:|:---:|:---:|
| 보존분 | $f \odot c_{t-1}$ | 0.620106 × 0.4 | 0.248043 |
| 반영분 | $i \odot g$ | 0.605874 × (−0.197375) | −0.119585 |
| 새 셀 상태 | $c_t$ | 0.248043 + (−0.119585) | **0.128458** |
| 새 은닉 상태 | $h_t = o \odot \tanh(c_t)$ | 0.485004 × 0.127756 | **0.061962** |

- 다음 시점은 $x_{t+1}=-0.3$이고 같은 가중치를 사용
- 이전 은닉 상태는 다음 가중합에, 이전 셀 상태는 다음 보존분에 들어감
- 결과: 새 셀 상태 $0.231282$, 새 은닉 상태 $0.103963$

## 2. 네 가중합 묶기

- PyTorch는 네 경로의 가중치를 쌓아 한 번에 계산, 쌓는 순서는 $i$ → $f$ → $g$ → $o$
- `weight_ih`: $(4H,\ F)$, `weight_hh`: $(4H,\ H)$, `bias_ih`, `bias_hh`: $(4H,)$
- 묶음 가중합 $= x_tW_{ih}^{T} + b_{ih} + h_{t-1}W_{hh}^{T} + b_{hh}$, 모양 $(B,\ 4H)$
- 마지막 축을 4등분(`chunk`)해 경로별 가중합 $(B,\ H)$ 네 개를 얻음
- 위 예시의 묶음 결과: $[[0.43,\ 0.49,\ -0.2,\ -0.06]]$

📐[공식]

$$\text{파라미터 수} = 4\,(H\times F + H\times H + 2H)$$

📍[예시] `nn.LSTMCell(2, 3)` ($F=2,\ H=3$)

| `weight_ih` $(12,\ 2)$ | `weight_hh` $(12,\ 3)$ | `bias_ih` $(12,)$ | `bias_hh` $(12,)$ | 합계 |
|:---:|:---:|:---:|:---:|:---:|
| 24 | 36 | 12 | 12 | **84** |

## 3. 게이트 한 항의 효과

- 망각 게이트 $f$: 이전 셀 상태의 보존분만 변경, 변화량 $\Delta c = (f_{\text{새}} - f_{\text{기준}}) \times c_{t-1}$
- 입력 게이트 $i$: 새 후보의 반영분만 변경
- 출력 게이트 $o$: 현재 은닉 상태만 변경, 셀 상태는 그대로
- 변한 은닉 상태는 다음 시점의 네 가중합에 들어가 다음 계산에 영향
- 이전 셀 상태가 음수이면 $f$를 줄일 때 $c$가 오히려 커질 수 있음

📍[예시] $c_{t-1}=0.5,\ i=0.3,\ g=0.4$ (반영분 0.12)

| 망각 게이트 $f$ | 보존분 $f \odot c_{t-1}$ | 새 셀 상태 $c_t$ | 기준 대비 |
|:---:|:---:|:---:|:---:|
| **0.6 (기준)** | 0.3 | **0.42** | |
| 0.2 | 0.1 | **0.22** | −0.2 |
| 0 | 0 | 0.12 | −0.3 |
| 1 | 0.5 | 0.62 | +0.2 |

- $c_t=0.42$에서 출력 게이트만 변경: $o=0.6$이면 $h_t\approx0.2382$, $o=0.2$이면 $h_t\approx0.0794$, 셀 상태는 0.42로 동일

## 4. 💻 [코드]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/rnn_practice.ipynb)

```python
import torch                                                   # 파이토치
from torch import nn                                           # 신경망 모듈

cell = nn.LSTMCell(1, 1).double()                              # 입력 1칸, 상태 1칸, float64
with torch.no_grad():                                          # 값 설정은 미분 기록에 넣지 않음
    cell.weight_ih.copy_(torch.tensor([[0.7], [0.9], [-0.6], [0.4]], dtype=torch.float64))  # 입력 가중치 (4, 1)
    cell.weight_hh.copy_(torch.tensor([[-0.4], [0.3], [0.5], [0.8]], dtype=torch.float64))  # 상태 가중치 (4, 1)
    cell.bias_ih.copy_(torch.tensor([0.0, 0.1, 0.2, -0.1], dtype=torch.float64))            # 입력 편향 (4,)
    cell.bias_hh.fill_(0.0)                                    # 상태 편향은 모두 0

x = torch.tensor([[0.5]], dtype=torch.float64)                 # 현재 입력 (1, 1)
h = torch.tensor([[-0.2]], dtype=torch.float64)                # 이전 은닉 상태 (1, 1)
c = torch.tensor([[0.4]], dtype=torch.float64)                 # 이전 셀 상태 (1, 1)

a = x @ cell.weight_ih.T + cell.bias_ih + h @ cell.weight_hh.T + cell.bias_hh  # 네 가중합 묶음 (1, 4)
ai, af, ag, ao = a.chunk(4, dim=1)                             # 네 칸으로 나눔, 각 (1, 1)
i, f, o = ai.sigmoid(), af.sigmoid(), ao.sigmoid()             # 입력·망각·출력 게이트
g = ag.tanh()                                                  # 후보
c_new = f * c + i * g                                          # 새 셀 상태 = 보존분 + 반영분
h_new = o * c_new.tanh()                                       # 새 은닉 상태

h_api, c_api = cell(x, (h, c))                                 # nn.LSTMCell이 계산한 결과
print(round(c_new.item(), 6), round(h_new.item(), 6))          # 손계산의 셀·은닉 상태
print(round(c_api.item(), 6), round(h_api.item(), 6))          # LSTMCell의 셀·은닉 상태
```

```
0.128458 0.061962
0.128458 0.061962
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
