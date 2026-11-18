---
title: "RNN - 04. 손실과 BPTT"
date: 2026-11-18 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, rnn, 손실, bptt, 역전파, 기울기, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. 손실과 학습 4단계

- 손실 $L$: 예측과 정답의 차이를 숫자로 만든 값, 클수록 나쁨
- 제곱으로 부호를 없애고 차이가 클수록 크게 반영
- 앞의 $\tfrac12$은 미분할 때 생기는 2를 없애려는 것, 표준 MSE의 절반 

📐[공식]

$$L = \tfrac12(\hat y - y)^2$$

- $\hat y$: 예측값, $y$: 정답
- 예: $\hat y=0.8,\ y=0.5$이면 $L=\tfrac12\times0.3^2=0.045$

| 단계 | 하는 일 | 가중치 변경 |
|:---:|:---:|:---:|
| forward | 입력에서 예측까지 계산 | 없음 |
| loss | 예측과 정답의 차이를 $L$로 계산 | 없음 |
| `backward()` | 각 가중치의 기울기 계산 | **없음** |
| `optimizer.step()` | 기울기로 가중치 갱신 | **있음** |

## 2. 변화율과 연쇄법칙

- 기울기 $\partial L/\partial w$: 가중치 $w$를 조금 늘렸을 때 손실이 변하는 비율
- 가중치는 기울기의 **반대 방향**으로 바꿈 (기울기가 양수이면 줄임)
- 연쇄법칙: 변화가 여러 단계를 지나면 단계별 변화율을 **곱함**
- 예: $\partial L/\partial \hat y = \hat y - y = 0.3$, $\hat y$를 0.01 늘리면 $L$은 0.045에서 약 0.048로 증가

📐[공식]

$$h=\tanh(a) \;\Rightarrow\; \frac{\partial h}{\partial a} = 1-h^2$$

- $h$가 $\pm1$에 가까워 포화될수록 변화율이 0에 가까워짐

## 3. BPTT와 공유 가중치의 기울기

- BPTT(Backpropagation Through Time): 시간으로 펼친 RNN 계산에 역전파를 적용
- 같은 $W_{hh}$를 모든 시점에서 사용하므로, 사용한 곳마다의 기여를 **더함**
- 한 경로에서는 변화율을 **곱하고**, 같은 가중치의 여러 사용 위치는 **더함**
- $\delta_t$: 시점 $t$의 가중합 $a_t$가 1 늘 때 손실이 늘어나는 양 ($\partial L/\partial a_t$)
- 아래 식은 $H=1$이고 마지막 상태 $h_T$를 그대로 예측으로 쓰는 경우

📐[공식]

$$\delta_T = (h_T - y)(1-h_T^2), \qquad \delta_t = \delta_{t+1}\,W_{hh}\,(1-h_t^2)$$

$$\frac{\partial L}{\partial W_{hh}} = \sum_t \delta_t\,h_{t-1}$$

- $W_{hh}$: 이전 상태를 현재 상태로 전달하는 가중치 (hidden → hidden)
- 시점 $t$에서 $W_{hh}$에 곱해진 값은 $h_{t-1}$이므로 기여는 $\delta_t h_{t-1}$

📍[예시] $x=[2,\ 0,\ 1],\ W_{ih}=0.5,\ W_{hh}=0.4$, 편향 없음, $h_0=0,\ y=0.3$

- 순전파: $h_1=0.761594,\ h_2=0.295551,\ h_3=\hat y=0.549888$, 손실 $L=0.031222$

| 시점 $t$ | $h_{t-1}$ | $h_t$ | $1-h_t^2$ | $\delta_t$ | 기여 $\delta_t h_{t-1}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 3 | 0.295551 | 0.549888 | 0.697623 | 0.249888×0.697623 = 0.174328 | 0.051523 |
| 2 | 0.761594 | 0.295551 | 0.912650 | 0.174328×0.4×0.912650 = 0.063640 | 0.048468 |
| 1 | 0 | 0.761594 | 0.419974 | 0.063640×0.4×0.419974 = 0.010691 | 0 |
| 합 | | | | | **0.099991** |

- 시점 1의 기여가 0인 것은 이 예제의 $h_0=0$ 때문, 항상 0은 아님
- 입력 가중치 $W_{ih}$는 곱해진 값이 $x_t$이므로 기여가 $\delta_t x_t$, 합은 0.195709

## 4. 💻 [코드]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/rnn_practice.ipynb)

```python
import torch                                                    # 파이토치
from torch import nn                                            # 신경망 모듈

rnn = nn.RNN(1, 1, bias=False, batch_first=True).double()       # 입력 1칸, 상태 1칸, 편향 없음
with torch.no_grad():                                           # 값 설정은 미분 기록에 넣지 않음
    rnn.weight_ih_l0.fill_(0.5)                                 # W_ih = 0.5
    rnn.weight_hh_l0.fill_(0.4)                                 # W_hh = 0.4

x = torch.tensor([[[2.0], [0.0], [1.0]]], dtype=torch.float64)  # 입력, 모양 (1, 3, 1)
y = torch.tensor([[0.3]], dtype=torch.float64)                  # 정답, 모양 (1, 1)

output, h_n = rnn(x)                                            # 모든 시점의 상태
pred = output[:, -1, :]                                         # 마지막 상태를 예측으로 사용
loss = 0.5 * ((pred - y) ** 2).sum()                            # 손실 L = ½(예측 − 정답)²
loss.backward()                                                 # 기울기 계산 (가중치는 바뀌지 않음)

print(round(loss.item(), 6))                                    # 손실 값
print(round(rnn.weight_hh_l0.grad.item(), 6), round(rnn.weight_ih_l0.grad.item(), 6))  # W_hh, W_ih의 기울기
```

```
0.031222
0.099991 0.195709
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
