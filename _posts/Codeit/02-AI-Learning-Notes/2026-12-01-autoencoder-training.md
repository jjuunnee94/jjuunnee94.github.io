---
title: "AutoEncoder - 08. 훈련 반복과 평가"
date: 2026-12-01 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, training-loop, adam, evaluation, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 훈련 반복
- 배치(B): 이미지를 한 번에 묶어 처리하는 장 수, 한 배치마다 가중치를 한 번 갱신
- epoch(E): 훈련 이미지 전체를 한 바퀴 도는 것
- 강의 설정: 훈련 5,000장, 배치 64장, 3 epoch, Adam 학습률 0.01
- 한 배치의 한 단계
  1. 이 배치에 새 노이즈를 만들어 noisy 생성
  2. 순전파로 복원값 계산
  3. 복원값과 clean의 평균제곱오차 계산
  4. 기울기 초기화 (`zero_grad`): 파이토치는 기울기를 쌓으므로 배치마다 0으로 시작 (슬라이드에 없음)
  5. 역전파 (`backward`)
  6. 가중치 갱신 (`step`)
- Adam: `SGD`와 같은 갱신 도구로 `step()` 사용법은 같음 (내부 원리는 슬라이드에 없음)

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 배치 수 | 올림(훈련 장 수 ÷ 배치 크기) | $\lceil N/B \rceil$ |
| 갱신 횟수 | 배치 수 × epoch 수 | $\lceil N/B \rceil \times E$ |
| 개선량 | 기준선 오차 − 복원 오차 | $\Delta=\text{MSE}_{base}-\text{MSE}_{rec}$ |

- 올림은 소수점이 있으면 1을 올림, 마지막 배치는 장 수가 모자라도 한 번 갱신
- 강의 계산: ⌈5000/64⌉ = ⌈78.125⌉ = 79, 79×3 = **237**

📍[예시] 갱신 횟수: 훈련 1,000장, 배치 64, 5 epoch

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 나누기 | 1000 ÷ 64 | 15.625 |
| 올림 | ⌈15.625⌉ | **16** (배치 수) |
| 마지막 배치 장 수 | 1000 − 15×64 | **40장** |
| 갱신 횟수 | 16 × 5 | **80** |

## 2. 평가
- 학습에 쓰지 않은 고정 검증 이미지(강의 10,000장)에 같은 노이즈 입력으로 비교
- 기준선: noisy ↔ clean 오차, 모델 점수: 복원값 ↔ clean 오차
- Δ가 양수면 개선, 음수면 악화, 0이면 동률
- 강의 결과: 학습 전 0.2256(무작위 초기 가중치) → 학습 후 0.0078, 기준선 0.0327, 개선 10,000 / 악화 0 / 동률 0

📍[예시] 개선량: 검증 3장

| 이미지 | 기준선 MSE | 복원 MSE | Δ = 기준선 − 복원 | 판정 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.040 | 0.010 | **0.030** | 개선 |
| 2 | 0.032 | 0.012 | **0.020** | 개선 |
| 3 | 0.050 | 0.055 | **−0.005** | 악화 |
| 평균 | 0.0407 | 0.0257 | | 개선 2 / 악화 1 / 동률 0 |

- 평균이 좋아져도 악화된 이미지가 있을 수 있어 이미지별 개선/악화 개수를 함께 봄

## 3. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
def train_step(net, optimizer, noisy_images, clean_images):  # 한 배치를 학습하는 함수
    net.train()  # 모델을 학습 모드로
    optimizer.zero_grad()  # 쌓인 기울기를 0으로 초기화
    reconstructed = net(noisy_images)  # 순전파: 노이즈 입력 → 복원값
    error = nn.MSELoss()(reconstructed, clean_images)  # 복원값과 clean의 평균제곱오차
    error.backward()  # 역전파: 기울기 계산
    optimizer.step()  # 가중치 갱신
