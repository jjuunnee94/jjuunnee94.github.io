---
title: "AutoEncoder - 03. 경사 하강법과 파라미터 갱신"
date: 2026-11-26 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, gradient-descent, learning-rate, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 경사 하강법 
- 한 바퀴: 순전파 → 손실 → 역전파 → 모든 가중치 한꺼번에 갱신
- 기울기가 음수면 가중치 증가, 양수면 감소
- 학습률(η): 이동량의 크기 조절, 크면 지나칠 수 있음 

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 가중치 갱신 | 새 가중치 = 현재 가중치 − 학습률 × 기울기 | $\theta_{new}=\theta-\eta\frac{\partial L}{\partial\theta}$ |

📍[예시] 앞 편 기울기 사용, $\eta=0.1$

| 가중치 | 현재 값 | 이동량(η×기울기) | 갱신 값 |
|:---:|:---:|:---:|:---:|
| $e_1$ | 0.5 | 0.1×(−3) = −0.3 | 0.5−(−0.3) = **0.8** |
| $e_2$ | 0.2 | 0.1×(−4.5) = −0.45 | 0.2−(−0.45) = **0.65** |
| $d_1$ | 1.0 | 0.1×(−0.64) = −0.064 | 1.0−(−0.064) = **1.064** |
| $d_2$ | 0.5 | 0.1×(−3.52) = −0.352 | 0.5−(−0.352) = **0.852** |

## 2. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
import torch  # 파이토치
import torch.nn as nn  # 신경망 층 모음

encoder = nn.Linear(2, 1, bias=False)  # 인코더: 2칸 → 1칸
decoder = nn.Linear(1, 2, bias=False)  # 디코더: 1칸 → 2칸
encoder.weight = nn.Parameter(torch.tensor([[0.5, 0.2]]))  # 인코더 가중치(e) 지정
decoder.weight = nn.Parameter(torch.tensor([[1.0], [0.5]]))  # 디코더 가중치(d) 지정
model = nn.Sequential(encoder, decoder)  # 인코더 → 디코더 연결
x = torch.tensor([[2.0, 3.0]])  # 입력(x)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 경사 하강법, 학습률(η)=0.1

loss = nn.MSELoss()(model(x), x)  # 순전파 후 평균제곱오차(L)
loss.backward()  # 역전파: 모든 기울기 계산
print(encoder.weight.grad, decoder.weight.grad)  # 인코더·디코더 가중치 기울기
optimizer.step()  # 가중치 갱신
print(encoder.weight.data, decoder.weight.data)  # 갱신된 가중치
```

```
tensor([[-3.0000, -4.5000]]) tensor([[-0.6400],
        [-3.5200]])
tensor([[0.8000, 0.6500]]) tensor([[1.0640],
        [0.8520]])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
