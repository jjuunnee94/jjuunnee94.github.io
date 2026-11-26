---
title: "이미지 변환 흐름 - Pillow 이미지 → NumPy 배열 → PyTorch 텐서"
date: 2026-11-26 15:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, cv, pillow, numpy, pytorch, tensor, dtype, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 변환 흐름

```
[ 하드디스크 사진.jpg ]
       │
       ▼  1단계: 파일 읽기 / 기본 편집
[ Pillow 이미지 (PIL) ]  ──▶ 사진 열기, 크기 조절, 좌우 반전에 최적화
       │
       ▼  2단계: 숫자 행렬로 변환 (OpenCV는 바로 여기로 읽음)
[ NumPy 배열 (ndarray) ] ──▶ 픽셀 숫자 직접 수정, 그래프(Matplotlib) 출력에 최적화
       │
       ▼  3단계: 딥러닝 모델 입력용으로 변환 (v2.ToImage() 등)
[ PyTorch 텐서 (Tensor) ] ──▶ GPU 고속 병렬 연산, 역전파(미분) 학습에 최적화
```

- 같은 사진을 담는 **서로 다른 그릇**이고, 그릇마다 잘하는 일이 달라서 갈아탄다.
- 갈아탈 때 바뀌는 것은 **축 순서, 숫자 종류, 숫자 범위** 세 가지뿐이다.y
## 2. 세 그릇 비교

- 아래는 가로 640, 세로 480인 사진 기준이다.

| 단계 | 그릇 | 모양 | 숫자 종류 | 숫자 범위 | 잘하는 일 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 필로우 이미지(PIL) | (가로, 세로) = (640, 480) | 사진 그 자체 | - | 열기, 크기 조절, 반전 |
| 2 | 넘파이 배열(NumPy) | (세로, 가로, 색) = (480, 640, 3) | 정수(uint8) | **0~255** | 픽셀 값 수정, CPU 계산 |
| 3 | 파이토치 텐서(Tensor) | (색, 세로, 가로) = (3, 480, 640) | 실수(float32) | **0~1** | GPU 계산, 학습 |

- PyTorch는 **가로가 먼저**, Numpy는 **세로가 먼저**다.
- PyTorch는 **색(채널)이 맨 앞**에 온다.

## 3. 도구별 역할

| 도구 | 하는 일 | 계산 장소 | 미분(학습) |
|:---:|:---:|:---:|:---:|
| 넘파이(NumPy) | 숫자 표 계산 | CPU | 못 함 |
| 파이토치(PyTorch) | 숫자 표 계산 + 학습 | CPU 또는 GPU | **자동으로 함** |
| 맷플롯립(Matplotlib) | 그래프로 보여주기 | - | - |

- Numpy도 계산 도구이고, 그래프를 그리는 것은 Matplotlib이다.
- Matplotlib이 **Numpy 모양(세로, 가로, 색)**을 기대해서 그림을 볼 때만 Numpy로 돌아온다.
- 오가는 이유는 계산 도구가 달라서가 아니라 **모양 약속이 달라서**다.

## 4. 숫자 하나로 따라가기

- 📐[공식] 0~1 값 = 정수 값 ÷ 255 / $x' = \dfrac{x}{255}$
- 픽셀 하나의 빨강 값이 128인 경우다.

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| Numpy | 그대로 | 128 (정수) |
| `ToImage` 후 | 축 순서만 바뀜 | 128 (**아직 정수**) |
| `ToDtype` 후 | $128 \div 255$ | 약 **0.502** (실수) |

- 📌[참고] 슬라이드에 없음: `v2.ToImage()`는 축 순서만 바꾸고, 0~1 실수로 바꾸는 일은 `v2.ToDtype(..., scale=True)`가 따로 한다.
- 세로 2, 가로 3인 사진의 모양 변화는 다음과 같다.

| 그릇 | 모양 | 의미 |
|:---:|:---:|:---:|
| Numpy | (2, 3, 3) | 세로 2, 가로 3, 색 3 |
| 텐서 | (3, 2, 3) | 색 3, 세로 2, 가로 3 |

## 5. 되돌아올 때 (PyTorch → Numpy)

- Matplotlib으로 보려면 축 순서를 (색, 세로, 가로) → (세로, 가로, 색)으로 되돌린다.
- `permute(1, 2, 0)`은 지금 1번 축을 맨 앞, 2번 축을 그다음, 0번 축을 맨 뒤로 옮긴다는 뜻이다.
- 값은 여전히 0~1 실수이고, Matplotlib은 0~1 실수도 그릴 수 있어서 그대로 보인다.

## 6. 💻 [코드]

```python
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import v2

img = Image.new("RGB", (3, 2), (128, 0, 255))  # 가로 3, 세로 2, 모든 픽셀이 (128, 0, 255)인 Pillow 이미지
print(img.size)                                # PyTorch는 (가로, 세로) 순서

arr = np.array(img)                            # Numpy 배열로 변환
print(arr.shape, arr.dtype)                    # Numpy는 (세로, 가로, 색) 순서, 정수

tf = v2.Compose([                              # 변환을 순서대로 묶기
    v2.ToImage(),                              # 텐서로 변환, 축 순서만 (색, 세로, 가로)로 바뀜
    v2.ToDtype(torch.float32, scale=True),     # 실수로 바꾸고 255로 나눔
])
tensor = tf(img)                               # PyTorch 이미지를 텐서로 변환
print(tensor.shape, tensor.dtype)              # 텐서는 (색, 세로, 가로) 순서, 실수

back = tensor.permute(1, 2, 0).numpy()         # (색, 세로, 가로) → (세로, 가로, 색), Numpy로 변환
print(back.shape)                              # Matplotlib이 기대하는 모양
```

```
(3, 2)
(2, 3, 3) uint8
torch.Size([3, 2, 3]) torch.float32
(2, 3, 3)
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
