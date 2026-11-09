---
title: "CNN - 09. 학습 루프, CrossEntropyLoss"
date: 2026-11-13 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, cnn, crossentropyloss, 학습루프, epoch, optimizer, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. 데이터, 점수, 정답의 대응

| 대상 | 모양 | 설명 |
|:---:|:---:|:---:|
| images | `[N,1,28,28]` | 입력 이미지 |
| logits | `[N,10]` | 모델이 계산한 클래스 점수 |
| labels | `[N]` | 데이터가 주는 정답 (0~9 정수) |
| loss | 스칼라 1개 | 점수와 정답의 차이 |

- 모델은 이미지만 입력받음, labels는 logits에서 만들어지는 값이 아님
- 같은 샘플 번호끼리 대응 (logits의 행 = 이미지, 열 = 클래스)

## 2. CrossEntropyLoss

- 입력 : raw logits `[N,10]` + 정수 labels `[N]`
- 내부에서 log-softmax를 처리하므로 모델 끝에 Softmax를 넣지 않음
- 샘플 하나의 손실은 정답 클래스 확률에 $-\ln$을 취한 값
- $z$ : 점수(logits), $y$ : 정답 클래스, $p_y$ : 정답 클래스의 확률
- N장이면 이미지별 손실을 평균(mean)해서 스칼라 1개로 만듦
- loss는 정확도(%)와 다른 값

📍 **[예시]** 점수 $[2, 1, 0]$

| 정답 | $p_y$ | $\ell=-\ln p_y$ |
|:---:|:---:|:---:|
| 클래스 0 | $e^2/(e^2+e^1+e^0) = 0.665$ | 0.408 |
| 클래스 2 | $e^0/(e^2+e^1+e^0) = 0.090$ | 2.408 |

- 점수가 모두 같으면 $p_y = 1/K$ → 손실 $\ln K$ (클래스 3개면 $\ln 3 \approx 1.099$)

## 3. 한 배치의 학습 5단계

- `criterion = nn.CrossEntropyLoss()` : 손실 함수 객체
- `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)` : 갱신할 파라미터와 방법, `lr`은 갱신 크기

| 순서 | 코드 | 하는 일 |
|:---:|:---|:---|
| 1 | `optimizer.zero_grad()` | 이전 배치의 grad 비우기 |
| 2 | `logits = model(images)` | 현재 가중치로 점수 계산 |
| 3 | `loss = criterion(logits, labels)` | 정답과 비교한 배치 평균 손실 |
| 4 | `loss.backward()` | 각 파라미터의 grad 계산 |
| 5 | `optimizer.step()` | grad로 가중치 갱신 |

- grad : 파라미터를 조금 바꿀 때 loss가 얼마나, 어느 방향으로 변하는지
- `backward()`는 grad를 저장만 하고, 가중치는 `step()`에서 바뀜
- `backward()`는 grad를 더하므로 `zero_grad()`가 필요 (가중치를 0으로 만드는 것이 아님)

## 4. epoch와 step

- step : 배치 하나로 가중치를 한 번 갱신
- epoch : 훈련 데이터 전체를 한 번 다 읽는 범위

📍 **[예시]** 이미지 1,000장, `batch_size=64`, `drop_last=False`

| | 계산 | 값 |
|:---:|:---:|:---:|
| 완전한 배치 | $\lfloor 1000/64 \rfloor = 15$개 | $15\times64=960$장 |
| 마지막 배치 | $1000-960$ | 40장 |
| step 수 | $15+1$ | **16 step** |

- `drop_last=False`면 남은 이미지도 마지막 배치로 사용

## 5. 훈련 평균 loss

- 배치 loss는 그 배치의 평균 → 마지막 배치가 작으면 단순 평균은 이미지 비중이 달라짐
- 손실 합 = 배치 평균 loss × 이미지 수 → 전체 손실 합 ÷ 전체 이미지 수

📍 **[예시]** 배치 A(64장, loss 0.5), 배치 B(40장, loss 0.8)

| 방식 | 계산 | 결과 |
|:---:|:---:|:---:|
| 단순 평균 | $(0.5+0.8)/2$ | 0.65 |
| 이미지 수 가중 평균 | $(64\times0.5 + 40\times0.8)/(64+40)$ | **0.615** |

- 훈련 중 서로 다른 갱신 시점의 loss를 누적한 값 (훈련 후 최종 모델로 다시 평가한 loss와 다름)
- 학습 셀을 다시 실행하면 새 모델로 1 epoch를 다시 시작 (이어학습이 아님)

## 6. 💻 [코드]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cnn_practice.ipynb)

```python
import torch
from torch import nn

criterion = nn.CrossEntropyLoss()
logits = torch.tensor([[2., 1., 0.]])             # 이미지 1장, 클래스 3개 점수
print(criterion(logits, torch.tensor([0])))       # 정답 0 : 정답 점수가 가장 큼
print(criterion(logits, torch.tensor([2])))       # 정답 2 : 정답 점수가 가장 작음

same = torch.zeros(1, 3)                          # 점수가 모두 같음
print(criterion(same, torch.tensor([1])))         # ln 3

two = torch.tensor([[2., 1., 0.], [2., 1., 0.]])  # 이미지 2장
print(criterion(two, torch.tensor([0, 2])))       # 두 손실의 평균

print(1000 // 64, 1000 % 64)                      # 완전한 배치 수, 남은 이미지 수
print((64 * 0.5 + 40 * 0.8) / (64 + 40))          # 이미지 수 가중 평균 loss
```

```
tensor(0.4076)
tensor(2.4076)
tensor(1.0986)
tensor(1.4076)
15 40
0.6153846153846154
```

```python
model = SimpleCNN()                                   # 새 모델 (초기값에서 시작)
criterion = nn.CrossEntropyLoss()                     # 손실 함수 객체
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_sum, seen = 0.0, 0                               # 손실 합, 지금까지 본 이미지 수
for images, labels in train_loader:                   # 배치 하나씩 꺼냄
    optimizer.zero_grad()                             # 1. grad 비우기
    logits = model(images)                            # 2. 점수 계산
    loss = criterion(logits, labels)                  # 3. 손실 계산
    loss.backward()                                   # 4. grad 계산
    optimizer.step()                                  # 5. 가중치 갱신
    loss_sum += loss.item() * labels.size(0)          # 배치 평균 loss × 배치 이미지 수
    seen += labels.size(0)                            # 이미지 수 누적
print(loss_sum / seen)                                # 이미지 수 가중 평균 loss
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
