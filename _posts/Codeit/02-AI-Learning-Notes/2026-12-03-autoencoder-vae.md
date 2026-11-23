---
title: "AutoEncoder - 10. VAE"
date: 2026-12-03 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, vae, kl-divergence, reparameterization, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. VAE(변분 오토인코더)
- 일반 AE는 인코더가 z를 직접 정하고, VAE는 인코더가 평균(μ)과 표준편차(σ)를 내놓아 정해진 규칙(분포)에서 z를 뽑음
- 평균(μ): 뽑히는 값의 중심, 표준편차(σ): 흩어지는 정도, 잠재 좌표마다 두 값이 있음
- 입력이 같아도 뽑힌 z가 달라져 복원이 조금씩 달라짐
- 입력 없이 새 이미지를 만들려면 디코더에 넣을 z를 입력 없이 뽑을 수 있어야 함
- prior: 생성할 때 z를 뽑는 공통 분포, 강의에서는 중심 0, 흩어짐 1인 표준정규분포 N(0, I)
- 입력별 분포(q)가 prior에서 너무 멀어지지 않도록 비용(KL 비용)을 줌
- 학습이 끝나면 prior에서 새 z를 뽑아 복원 때 쓴 같은 디코더에 넣어 입력 없이 생성
- 강의 구조: 완전연결, 잠재 16차원(z는 16칸, 평균·표준편차는 각각 16칸), 훈련 50,000장, 20 epoch

| 지금 나온 말 | 쉬운 뜻 | 이미 배운 것과 연결 |
|:---:|:---:|:---:|
| 재구성 비용 | 복원이 원본과 얼마나 다른가 | 평균제곱오차(MSE)와 같은 것 |
| KL 비용 | 입력별 분포가 prior와 얼마나 다른가 | Sparse AE의 벌점 자리에 들어가는 점수 |
| 총 손실 | 두 점수의 합 | 재구성 비용 + KL 비용 |

| | 일반 AE | VAE |
|:---:|:---:|:---:|
| 인코더가 내놓는 것 | z | 평균(μ), 표준편차(σ) |
| z | 인코더가 바로 정함 | μ, σ로 정해진 규칙에서 뽑음 |
| 손실 | 평균제곱오차 | 재구성 비용 + KL 비용 |

## 2. 재매개변수화와 식
- 뽑는 과정을 그대로 두면 역전파가 인코더까지 이어지지 않음
- 무작위는 바깥에서 만든 난수(ε)로 빼고 z를 μ와 σ로 계산하면 기울기가 인코더까지 이어짐

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 재매개변수화 | 잠재 표현 = 평균 + 표준편차 × 표준정규 난수 | $z=\mu+\sigma\odot\varepsilon$ |
| 총 손실 | 재구성 비용 + KL 비용 | $L=L_{recon}+\text{KL}$ |
| KL 비용 (가우시안일 때, 슬라이드에 없음) | −½ × (1 + log 분산 − 평균² − 분산)의 합 | $-\frac12\sum\left(1+\log\sigma^2-\mu^2-\sigma^2\right)$ |

- ⊙는 칸끼리 곱, log는 자연로그, 분산은 σ², 평균 0·표준편차 1이면 KL = 0 (입력별 분포가 prior와 같음)

📍[예시] 평균 μ = (0.5, −1.0), 표준편차 σ = (2.0, 0.5), 난수 ε = (0.4, −1.2)

| 단계 | 첫째 칸 | 둘째 칸 |
|:---:|:---:|:---:|
| z = μ + σ × ε | 0.5 + 2.0×0.4 = **1.3** | −1.0 + 0.5×(−1.2) = **−1.6** |
| 분산 σ² | 4 | 0.25 |
| 1 + log σ² − μ² − σ² | 1 + 1.386 − 0.25 − 4 = −1.864 | 1 − 1.386 − 1 − 0.25 = −1.636 |
| KL (× −½) | **0.932** | **0.818** |

| 합계 | 계산 | 결과 |
|:---:|:---:|:---:|
| KL 비용 | 0.932 + 0.818 | **1.75** |
| 총 손실 (재구성 비용 0.06 가정) | 0.06 + 1.75 | **1.81** |

📍[예시] 한 칸의 KL 값이 변하는 모습

| 중심(μ) | 흩어짐(σ) | KL | 해석 |
|:---:|:---:|:---:|:---:|
| 0 | 1 | **0** | prior와 같음 |
| 1 | 1 | 0.5 | 중심이 0에서 벗어남 |
| 3 | 1 | 4.5 | 중심이 많이 벗어남 |
| 0 | 2 | 0.807 | 흩어짐이 1에서 벗어남 |
| 0 | 0.5 | 0.318 | 흩어짐이 1에서 벗어남 |

## 3. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
import torch  # 파이토치
import torch.nn as nn  # 신경망 층 모음

class VAE(nn.Module):  # 모델의 틀
    def __init__(self):  # 층 준비
        super().__init__()  # 물려받은 틀의 준비 작업
        self.enc = nn.Sequential(nn.Linear(784, 128), nn.ReLU())  # 인코더 앞부분
        self.mu = nn.Linear(128, 16)  # 평균(μ) 16칸을 만드는 층
        self.logvar = nn.Linear(128, 16)  # log 분산 16칸을 만드는 층
        self.dec = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())  # 디코더

    def forward(self, x):  # 순전파
        h = self.enc(x)  # 인코더 앞부분 통과
        mu, logvar = self.mu(h), self.logvar(h)  # 평균, log 분산
        std = torch.exp(0.5 * logvar)  # 표준편차(σ) = e^(0.5 × log 분산)
        eps = torch.randn_like(std)  # σ와 같은 모양의 표준정규 난수
        z = mu + std * eps  # 재매개변수화
        return self.dec(z), mu, logvar  # 복원값, 평균, log 분산

mu = torch.tensor([[0.5, -1.0]])  # 예시 평균(μ)
sigma = torch.tensor([[2.0, 0.5]])  # 예시 표준편차(σ)
eps = torch.tensor([[0.4, -1.2]])  # 예시 난수(ε)
z = mu + sigma * eps  # 재매개변수화로 z 계산
kl = -0.5 * (1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2).sum()  # KL 비용
print(z, kl)  # z, KL 출력
```

```
tensor([[ 1.3000, -1.6000]]) tensor(1.7500)
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
