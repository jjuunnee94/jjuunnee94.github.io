---
title: "AutoEncoder - 09. Sparse AE"
date: 2026-12-02 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, sparse-autoencoder, regularization, penalty, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. Sparse AE(희소 오토인코더)
- 입력을 복원하면서 잠재 표현(z)의 많은 값이 0 또는 0에 가깝도록 함께 학습
- 희소: 대부분 0이고 일부 칸만 주로 사용, 활성 = 0이 아닌 칸
- 학습 목표 두 가지: 복원 오차 줄이기, z의 크기(벌점) 줄이기
- 벌점 = 희소 규제 가중치(λ) × 평균 \|z\|, 절댓값을 써서 부호 상쇄를 막음
- 활성을 모두 0으로 만들면 입력을 구별할 정보를 잃으므로 복원도 함께 학습
- 활성값의 크기를 줄이는 방식이며 0이 아닌 값의 개수를 직접 세는 식이 아님
- 강의 구조: 완전연결, 잠재 64차원, 깨끗한 입력 복원 (노이즈 제거 모델과 직접 비교하지 않음), λ=0과 0.02 비교

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 희소 벌점 | 희소 규제 가중치 × 잠재 표현 절댓값의 평균 | $\lambda\,\text{mean}(\lvert z\rvert)$ |
| 총 손실 | 평균제곱오차 + 희소 벌점 | $L=\text{MSE}(\hat{x},x)+\lambda\,\text{mean}(\lvert z\rvert)$ |
| 활성 비율 | 0이 아닌 z의 개수 ÷ 전체 z의 개수 | $\frac{\text{0이 아닌 z의 개수}}{\text{전체 z의 개수}}$ |

📍[예시] 복원 평균제곱오차(MSE) 0.04, $\lambda=0.2$

| 단계 | A: 켜진 칸이 적은 z | B: 켜진 칸이 많은 z |
|:---:|:---:|:---:|
| 잠재 표현 z | (0, 1.5, 0, 0.5) | (1.0, 1.5, 0.5, 0.5) |
| 크기의 합 | 0+1.5+0+0.5 = 2.0 | 1+1.5+0.5+0.5 = 3.5 |
| 평균 \|z\| (÷4) | **0.5** | **0.875** |
| 벌점 (0.2 × 평균) | **0.1** | **0.175** |
| 총 손실 (0.04 + 벌점) | **0.14** | **0.215** |
| 활성 비율 | 2÷4 = **0.5** | 4÷4 = **1.0** |

- 복원 오차가 같으면 z가 희소한 쪽의 총 손실이 작아 학습이 그쪽으로 이끌림
- λ = 0이면 벌점이 없어 일반 오토인코더와 같음

## 2. 벌점이 활성을 줄이는 과정
- 벌점은 총 손실에 포함되어 역전파에서 복원 오차의 기울기와 함께 계산됨
- 벌점의 기울기는 z를 작게 하는 방향을 가리켜 가중치 갱신 때 반영되고, 다음 순전파의 z가 작아짐 (슬라이드에 없음)

| 순서 | 일어나는 일 | 벌점의 역할 |
|:---:|:---:|:---:|
| ① 순전파 | 입력 → z → 복원값 | 이 z로 벌점 계산 |
| ② 손실 계산 | 복원 오차 + λ × 평균 \|z\| | 총 손실에 포함 |
| ③ 역전파 | 두 기울기를 함께 구함 | z를 작게 하는 기울기 |
| ④ 가중치 갱신 | 기울기 반대 방향 이동 | z가 작아지는 쪽으로 이동 |
| ⑤ 다음 순전파 | 갱신된 가중치로 z 계산 | z가 전보다 작아짐 |

- z의 한 칸이 양수일 때 벌점의 기울기는 λ ÷ 칸 수 (n=4): λ=0.2는 0.05, λ=0.4는 0.10
- λ가 커지면 줄이려는 힘이 커지고, 복원 오차의 기울기와 균형을 이루는 곳에서 멈춤
- λ가 커지면 활성 비율과 복원 오차가 함께 바뀌므로 두 값을 같이 읽음

## 3. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
import torch  # 파이토치
import torch.nn as nn  # 신경망 층 모음

class SparseAE(nn.Module):  # 모델의 틀
    def __init__(self):  # 층 준비
        super().__init__()  # 물려받은 틀의 준비 작업
        self.encoder = nn.Sequential(nn.Linear(784, 64), nn.ReLU())  # 784칸 → 64칸, 음수는 0
        self.decoder = nn.Sequential(nn.Linear(64, 784), nn.Sigmoid())  # 64칸 → 784칸, 0~1 값

    def forward(self, x):  # 순전파
        z = self.encoder(x)  # 잠재 표현(z)
        return self.decoder(z), z  # 복원값과 z를 함께 돌려줌

# 학습 한 단계의 손실: rec, z = model(x)로 받아 아래처럼 계산
# loss = nn.MSELoss()(rec, x) + lam * z.abs().mean()

z = torch.tensor([[0.0, 1.5, 0.0, 0.5]])  # 예시 z (A)
mse_value = torch.tensor(0.04)  # 예시 복원 평균제곱오차
lam = 0.2  # 희소 규제 가중치(λ)
loss = mse_value + lam * z.abs().mean()  # 평균제곱오차 + λ × 평균 |z|
active = (z != 0).float().mean()  # 0이 아닌 칸의 비율
print(loss, active)  # 총 손실, 활성 비율
```

```
tensor(0.1400) tensor(0.5000)
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
