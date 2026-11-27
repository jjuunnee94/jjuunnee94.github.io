---
title: "컴퓨터 비전 - 10. 데이터 증강"
date: 2026-12-13 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, cv, computer-vision, data-augmentation, overfitting, torchvision, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 데이터 증강

- 데이터 증강(data augmentation): 학습 사진이 적어 과적합이 생길 때 같은 사진을 조금씩 바꿔 **매번 다른 학습 예**로 쓰는 방법, **라벨은 그대로**
- 전처리와 목적이 다름: 전처리 = 모델이 받는 형식 맞추기(크기, 값 범위), 증강 = 데이터를 다양하게 만들기
- 변형 사진을 디스크에 저장하지 않고 학습 중에 매번 새로 만듦
- AlexNet(2012): 256 × 256에서 224 × 224를 무작위로 자르고 뒤집어 학습 데이터가 "a factor of 2048"배가 되는 효과
- 판단 기준: **사람이 변환된 이미지를 보고 같은 라벨을 붙이는가?** → 라벨이 바뀌는 변환은 증강이 아님

| 상황 | 변환 | 사람이 붙이는 라벨 | 증강으로 쓸 수 있나 |
|:---:|:---:|:---:|:---:|
| 알파벳 b / d 분류 | 좌우 뒤집기 | b가 d처럼 보임 → 라벨이 바뀜 | **불가** |
| 꽃 사진 장미 / 튤립 분류 | 좌우 뒤집기 | 같은 꽃 | **가능** |
| 꽃 사진 장미 / 튤립 분류 | ±10° 회전 | 같은 꽃 | **가능** |

- 기하 증강(뒤집기·크롭·회전): 이미지와 라벨(박스·mask)에 같은 변환 / 색 증강(밝기·색조): 이미지에만

📍[예시] 도로 표지판 분류기에 좌우 반전과 색조 변환을 넣어도 될까?

| 변환 | 라벨이 바뀌는 사례 | 보존되는 사례 | 결론 |
|:---:|:---:|:---:|:---:|
| 좌우 반전 | 좌회전 표지판 → 우회전 표지판 | 좌우 대칭인 진입 금지 표지판 | 방향 표지판이 있으면 넣으면 안 됨 |
| 색조 변환 | 빨강 → 초록(색이 라벨인 표지판) | 흑백 속도 표지판 | 색이 라벨인 표지판이 있으면 넣으면 안 됨 |

## 2. 학습 경로와 평가 경로

- 무작위 증강은 **학습 경로에만** 넣음
- 평가 경로는 **언제나 같은 결과**: 평가 때도 무작위면 같은 모델·같은 데이터인데 점수가 매번 달라져서 비교할 수 없음

| 경로 | 변환 순서 |
|:---:|:---:|
| 학습 | `ToImage` → **`RandomResizedCrop(224, scale=(0.5, 1.0))`** → **`RandomHorizontalFlip()`** → `ToDtype(scale=True)` → `Normalize` |
| 평가 | `ToImage` → `Resize(256)` → `CenterCrop(224)` → `ToDtype(scale=True)` → `Normalize` |

- `RandomResizedCrop(224, scale=(0.5, 1.0))`: 원본 면적의 **50%~100%** 중 무작위 비율로 영역을 잘라 224 × 224로 만듦 (가로세로 비율도 기본적으로 조금씩 무작위로 바뀜)
- `RandomHorizontalFlip()`: 확률 p(기본 0.5)로 좌우 뒤집기

📍[예시] 원본이 200 × 300 = 60,000 픽셀이고 `scale=(0.5, 1.0)`일 때

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 원본 면적 | 200 × 300 | 60,000 |
| 최소 면적 (50%) | 60,000 × 0.5 | 30,000 |
| 최대 면적 (100%) | 60,000 × 1.0 | 60,000 |
| 잘라 내는 영역 | | **30,000~60,000** |

- 평가의 `Resize(256)` → `CenterCrop(224)`는 07편과 같은 방법
- 두 경로 모두 마지막에 같은 표준화를 해야 함 (09편)

## 3. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cv_practice.ipynb)

```python
import numpy as np                                           # 배열 라이브러리
import torch                                                 # PyTorch
from torchvision.transforms import v2                        # 이미지 변환 모음

MEAN = [0.485, 0.456, 0.406]                                 # 채널별 평균
STD = [0.229, 0.224, 0.225]                                  # 채널별 표준편차
to_float = v2.ToDtype(torch.float32, scale=True)             # float32로 바꾸고 ÷255

train_tf = v2.Compose([v2.ToImage(),
                       v2.RandomResizedCrop(224, scale=(0.5, 1.0)),   # 면적 50~100% 무작위 크롭
                       v2.RandomHorizontalFlip(),                      # 50% 확률 좌우 뒤집기
                       to_float, v2.Normalize(MEAN, STD)])             # 학습 경로
eval_tf = v2.Compose([v2.ToImage(),
                      v2.Resize(256), v2.CenterCrop(224),              # 항상 같은 결과
                      to_float, v2.Normalize(MEAN, STD)])              # 평가 경로

img = np.zeros((300, 451, 3), dtype=np.uint8)                # 높이 300, 너비 451인 사진
print(train_tf(img).shape, eval_tf(img).shape)               # 두 경로의 출력 모양
```

```
torch.Size([3, 224, 224]) torch.Size([3, 224, 224])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
