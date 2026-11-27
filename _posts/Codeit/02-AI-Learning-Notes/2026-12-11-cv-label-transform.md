---
title: "컴퓨터 비전 - 08. 라벨 함께 바꾸기"
date: 2026-12-11 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, cv, computer-vision, bounding-box, mask, tv-tensors, torchvision, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 기하 변환과 색 변환

- 라벨: 검출은 바운딩 박스, 세그멘테이션은 mask(픽셀마다 클래스 번호)
- 이미지만 바꾸고 라벨을 그대로 두면 오류 없이 라벨이 엉뚱한 곳을 가리킴

| 변환 종류 | 예 | 이미지 | 박스·mask |
|:---:|:---:|:---:|:---:|
| **기하 변환** | 뒤집기, 크롭, 회전, 리사이즈 | 적용 | **함께 적용** |
| **색 변환** | 밝기, 색조 | 적용 | 적용하지 않음 |

## 2. 박스 좌표 변환 (XYXY)

- 박스 = (x1, y1, x2, y2), 왼쪽 위 (x1, y1)과 오른쪽 아래 (x2, y2) / x는 1·3번째 값, y는 2·4번째 값
- 이미지의 너비(W), 높이(H)

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 좌우 뒤집기 | 새 x1 = 너비 W − 옛 x2, 새 x2 = 너비 W − 옛 x1, y는 그대로 | $x_1' = W - x_2,\; x_2' = W - x_1$ |
| 리사이즈 | 새 좌표 = 옛 좌표 × 배율 (x에는 가로 배율, y에는 세로 배율) | $x' = x \cdot s_x,\; y' = y \cdot s_y$ |

- 뒤집으면 왼쪽 점이 오른쪽 점이 되므로 x1과 x2가 서로 바뀜 → x1 < x2가 유지됨
- 가로 배율(sx) = 바꾼 너비(W′) ÷ 옛 너비(W), 세로 배율(sy) = 바꾼 높이(H′) ÷ 옛 높이(H)

📍[예시] 너비 W = 400, 높이 H = 300인 이미지에서 박스 (60, 45, 140, 120)을 좌우로 뒤집기

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 새 x1 | W − 옛 x2 = 400 − 140 | **260** |
| 새 x2 | W − 옛 x1 = 400 − 60 | **340** |
| y1, y2 | 그대로 | 45, 120 |
| 결과 | | **(260, 45, 340, 120)** |

📍[예시] 같은 이미지를 높이 H′ = 100, 너비 W′ = 200으로 리사이즈

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 가로 배율(sx) | 200 ÷ 400 | 0.5 |
| 세로 배율(sy) | 100 ÷ 300 | 1/3 |
| x1, x2 | 60 × 0.5, 140 × 0.5 | 30, 70 |
| y1, y2 | 45 × 1/3, 120 × 1/3 | 15, 40 |
| 결과 | | **(30, 15, 70, 40)** |

## 3. mask는 최근접으로

- mask의 숫자는 크기가 아니라 이름(클래스 번호) → 평균을 내면 없는 클래스가 생김

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 이웃 두 칸 | 클래스 0(배경), 클래스 3(눈) | |
| 쌍선형 평균 | (0 + 3) ÷ 2 | 1.5 → **클래스 1 또는 2** (원래 없던 다른 클래스) |
| 최근접 | 가까운 쪽 값을 복사 | **0 또는 3** (있던 클래스만) |

- torchvision v2는 `tv_tensors.Mask`에 쌍선형을 요청해도 최근접으로 처리
- `tv_tensors`: 이미지·박스·mask를 구분하는 자료형 → 변환에 함께 넣으면 각각 맞는 방식으로 적용

## 4. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cv_practice.ipynb)

```python
import numpy as np                                          # 배열 라이브러리
import torch                                                # PyTorch
from torchvision import tv_tensors                          # 이미지·박스·mask를 구분하는 자료형
from torchvision.transforms import v2                       # 이미지 변환 모음

img = np.zeros((300, 400, 3), dtype=np.uint8)               # 높이 300, 너비 400인 사진
boxes = tv_tensors.BoundingBoxes([(60, 45, 140, 120)],
                                 format="XYXY",              # (x1, y1, x2, y2) 형식
                                 canvas_size=(300, 400))     # 사진 크기 (높이, 너비)

flip = v2.RandomHorizontalFlip(p=1)                         # p=1: 항상 좌우로 뒤집기 (기본값 0.5)
flip_img, flip_boxes = flip(v2.ToImage()(img), boxes)       # 이미지와 박스를 함께 전달
print(flip_boxes.tolist())

small_img, small_boxes = v2.Resize((100, 200))(v2.ToImage()(img), boxes)   # (높이, 너비)로 리사이즈
print(small_boxes.tolist())
```

```
[[260, 45, 340, 120]]
[[30, 15, 70, 40]]
```

```python
label_row = torch.tensor([[[0, 0, 3, 3]]])                  # 모양 [1, 1, 4], 클래스 번호 4칸
mask = tv_tensors.Mask(label_row)                           # mask로 표시
out = v2.Resize((1, 8))(mask)                               # 너비 4 → 8
print(out[0, 0].tolist())                                   # 첫 채널의 첫 행
```

```
[0, 0, 0, 0, 3, 3, 3, 3]
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
