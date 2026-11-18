---
title: "RNN - 09. LSTM·GRU 비교, 입력 길이, 양방향 RNN"
date: 2026-11-23 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, rnn, lstm, gru, 양방향rnn, bidirectional, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. LSTM과 GRU 비교
- 반환값의 괄호는 **상태가 몇 개인지**를 그대로 반영
- 파라미터 수 식에서 $H$는 상태 칸 수, $F$는 입력 칸 수

| | LSTM | GRU |
|:---:|:---:|:---:|
| 상태 | 셀 상태 $c$, 은닉 상태 $h$ | 은닉 상태 $h$ |
| 게이트 | 입력 $i$, 망각 $f$, 출력 $o$ + 후보 $g$ | 리셋 $r$, 업데이트 $z$ + 후보 $n$ |
| 새 상태 | $c$를 갱신한 뒤 $h$를 꺼냄 | 후보를 이전 $h$와 섞음 |
| 반환값 | `output, (h_n, c_n)` | `output, h_n` |
| 파라미터 수 | $4(HF+H^2+2H)$ | $3(HF+H^2+2H)$ |

## 2. 입력 길이를 바꿔도 같은 모델
- 가중치는 한 시점을 처리하는 규칙 하나이고, 시점 수 $T$만큼 반복할 뿐
- $B$: 예제 수, $T$: 시점 수, $F$: 입력 칸 수, $H$: 상태 칸 수
- 바뀌는 것: `output`의 시점 축($T$)
- 안 바뀌는 것: `h_n`, `c_n`의 모양, 가중치의 모양 ($T$와 무관)
- 초기 상태를 주지 않으면 호출마다 처음 상태 0에서 시작
- 앞쪽 시점의 출력은 입력이 길어져도 같고, **마지막 상태 값**은 달라짐

📍[예시] 규칙 $h_t=\tanh(0.8\,x_t+0.5\,h_{t-1})$, 처음 상태 $h_0=0$, 입력 $[1,\ 2,\ 3]$

| 시점 | 계산 | 은닉 상태 |
|:---:|:---:|:---:|
| 1 | $\tanh(0.8\times1+0.5\times0)=\tanh(0.8)$ | **0.664037** |
| 2 | $\tanh(0.8\times2+0.5\times0.664037)=\tanh(1.932018)$ | **0.958896** |
| 3 | $\tanh(0.8\times3+0.5\times0.958896)=\tanh(2.879448)$ | **0.993711** |

- 입력을 $[1,\ 2]$만 넣으면 은닉 상태는 0.664037, 0.958896으로 앞 두 시점이 같고, 마지막 상태만 0.958896과 0.993711로 다름

## 3. 양방향 RNN
- 같은 입력창을 **앞→뒤**, **뒤→앞**으로 한 번씩 읽고 두 결과를 이어 붙임
- 정방향은 그 시점 이전을, 역방향은 그 시점 이후를 요약
- 방향마다 가중치가 따로여서 파라미터 수가 2배
- 인자: `bidirectional=True`

📍[예시] 위의 규칙으로 입력 $[1,\ 2,\ 3]$을 읽음 (설명을 위해 두 방향이 같은 규칙을 쓴다고 가정)

| 위치 | 정방향 (1→2→3) | 역방향 (3→2→1) | `output`의 한 줄 |
|:---:|:---:|:---:|:---:|
| 1 | 0.664037 | **0.857808** | [0.664037, 0.857808] |
| 2 | 0.958896 | 0.969973 | [0.958896, 0.969973] |
| 3 | **0.993711** | 0.983675 | [0.993711, 0.983675] |

- 정방향은 위치 3에서, 역방향은 위치 1에서 입력 전체를 읽은 상태가 됨

| 값 | 모양 | 뜻 |
|:---:|:---:|:---|
| `output` | ($B$, $T$, 2$H$) | 앞 $H$칸은 정방향, 뒤 $H$칸은 역방향 |
| `h_n`, `c_n` | (2, $B$, $H$) | `[0]` 정방향 마지막, `[1]` 역방향 마지막 |

- `h_n[0]`은 `output[:, -1, :H]`와, `h_n[1]`은 `output[:, 0, H:]`와 같음
- 파라미터 수: LSTM 한 방향이 $4(HF+H^2+2H)$이므로 양방향은 2배

⚠️[주의] 예측 시점에 아직 관측되지 않은 값(정답)은 입력창에 넣지 않는다.

## 4. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/rnn_practice.ipynb)

```python
import torch                                                   # 파이토치
from torch import nn                                           # 신경망 모듈

torch.manual_seed(0)                                           # 무작위 초기값 고정
lstm = nn.LSTM(1, 3, batch_first=True)                         # 입력 1칸, 상태 3칸
x4 = torch.tensor([[[1.], [2.], [3.], [4.]]])                  # 시점 4개, (1, 4, 1)
x6 = torch.tensor([[[1.], [2.], [3.], [4.], [5.], [6.]]])      # 시점 6개, (1, 6, 1)
out4, (h4, c4) = lstm(x4)                                      # 같은 모델에 4시점 입력
out6, (h6, c6) = lstm(x6)                                      # 같은 모델에 6시점 입력
print(out4.shape, out6.shape, h4.shape, h6.shape)              # 네 모양 비교
print(torch.allclose(out6[:, :4, :], out4))                    # 앞 4시점 출력이 같은지

bi = nn.LSTM(1, 2, batch_first=True, bidirectional=True)       # 양방향, 입력 1칸, 상태 2칸
x = torch.tensor([[[0.5], [-0.3], [0.8], [0.1]]])              # 시점 4개, (1, 4, 1)
output, (h_n, c_n) = bi(x)                                     # 앞→뒤, 뒤→앞으로 각각 계산
print(output.shape, h_n.shape, c_n.shape)                      # 세 모양
print(torch.allclose(output[:, -1, :2], h_n[0]),               # 마지막 위치 앞 2칸 = 정방향 마지막
      torch.allclose(output[:, 0, 2:], h_n[1]))                # 첫 위치 뒤 2칸 = 역방향 마지막
```

```
torch.Size([1, 4, 3]) torch.Size([1, 6, 3]) torch.Size([1, 1, 3]) torch.Size([1, 1, 3])
True
torch.Size([1, 4, 4]) torch.Size([2, 1, 2]) torch.Size([2, 1, 2])
True True
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
