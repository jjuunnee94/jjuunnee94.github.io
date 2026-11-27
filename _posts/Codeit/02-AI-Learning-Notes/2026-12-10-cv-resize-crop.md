---
title: "컴퓨터 비전 - 07. 크기 바꾸기와 자르기"
date: 2026-12-10 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, cv, computer-vision, resize, interpolation, crop, torchvision, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 크기를 맞추는 이유와 보간

- 이미지를 배치로 묶으려면 크기가 모두 같아야 함
- ImageNet으로 이미 학습한 모델은 학습 때 쓴 크기(예: 224 × 224)를 기대함
- 크기를 바꾸면 원래 격자에 없는 위치의 값이 필요 → 주변 픽셀로 추정하는 것이 보간(interpolation)

| 방식 | 하는 일 |
|:---:|:---:|
| **최근접(nearest)** | 가장 가까운 픽셀 값을 **복사** |
| **쌍선형(bilinear)** | 가까운 이웃 4개를 거리에 따라 **가중 평균** |

📍[예시] 위치 0에 값 40, 위치 1에 값 80일 때 위치 **0.25**의 값

| 방식 | 계산 | 결과 |
|:---:|:---:|:---:|
| 최근접 | 0.25는 위치 0에 더 가까움 → 40을 복사 | **40** |
| 쌍선형 | 위치 0까지 거리 0.25, 위치 1까지 거리 0.75 → 가까운 쪽에 큰 가중치 (0.75, 0.25) → 40 × 0.75 + 80 × 0.25 | **50** |

- 이웃 4개 = 위 계산을 가로와 세로로 두 번 하는 것

## 2. 라이브러리별 기본 설정

- 축소할 때는 06편의 에일리어싱을 막기 위해 흐리기(안티에일리어싱)가 필요

| 함수 | 기본 보간 | 축소 때 안티에일리어싱 |
|:---:|:---:|:---:|
| torchvision v2 `Resize` | bilinear | 기본 켜짐(`antialias=True`) |
| `torch.nn.functional.interpolate` | nearest | 기본 꺼짐 |
| OpenCV `cv2.resize` | `INTER_LINEAR` | 인자 없음 → 축소는 `INTER_AREA` 권장 |
| Pillow `Image.resize` | BICUBIC | bilinear·bicubic에서 늘 적용 |

- 기본값은 판마다 다를 수 있으므로 쓰는 판의 문서를 확인

## 3. Resize와 CenterCrop

- `Resize(정수)`: **짧은 변**을 그 크기에 맞추고 **비율 유지**, 긴 변은 같은 비율로 줄이며 소수점은 내림(floor)
- `Resize((높이, 너비))`: 두 변을 정해진 크기로 맞춤, **비율이 깨질 수 있음**

📍[예시] 높이(H) 200, 너비(W) 333인 이미지를 `Resize(100)`

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 짧은 변 | 200 → 100 | 높이 100 |
| 배율 | 100 ÷ 200 | 0.5 |
| 긴 변 | 333 × 0.5 = 166.5 → 내림 | 너비 **166** |
| 결과 모양 | | [3, 100, 166] |

- `Resize((100, 100))`은 [3, 100, 100]이고 비율이 깨져 세로로 눌려 보임
- `CenterCrop(100)`: 가운데 100 × 100만 남기고 바깥을 버림 → [3, 100, 166]에서 너비 66칸(166 − 100)을 양옆에서 33칸씩 버려 [3, 100, 100], 비율은 유지되고 양옆이 잘림

## 4. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cv_practice.ipynb)

```python
import torch                                              # PyTorch
from torchvision.transforms import v2                     # 이미지 변환 모음

x = torch.zeros(3, 200, 333, dtype=torch.uint8)           # 모양 [C=3, H=200, W=333] 이미지 텐서

print(v2.Resize(100)(x).shape)                            # 짧은 변을 100으로, 비율 유지
print(v2.Resize((100, 100))(x).shape)                     # (높이, 너비)를 둘 다 지정
print(v2.CenterCrop(100)(v2.Resize(100)(x)).shape)        # 줄인 뒤 가운데 100×100만 남김
small = v2.Resize((50, 83), antialias=False)(x)           # 줄이기 전에 흐리지 않는 설정
print(small.shape)
```

```
torch.Size([3, 100, 166])
torch.Size([3, 100, 100])
torch.Size([3, 100, 100])
torch.Size([3, 50, 83])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
