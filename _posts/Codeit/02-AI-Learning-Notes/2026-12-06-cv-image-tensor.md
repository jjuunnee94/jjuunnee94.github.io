---
title: "컴퓨터 비전 - 03. 이미지 텐서와 값 범위"
date: 2026-12-06 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, cv, computer-vision, tensor, permute, normalization, torchvision, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 축 순서

- 이미지 라이브러리(NumPy, OpenCV): **(높이 H, 너비 W, 채널 수 C)** = HWC
- PyTorch `Conv2d` 입력: **[이미지 수 N, 채널 수 C, 높이 H, 너비 W]** = NCHW
- 사진 한 장 (300, 451, 3) → `Conv2d`에는 **[1, 3, 300, 451]**
- 모양이 틀려도 오류 없이 지나가는 경우가 있어서 어떤 함수로 바꾸는지가 중요

📍[예시] 높이(H) = 2, 너비(W) = 3, 채널 수(C) = 2인 배열 `x = torch.arange(12).reshape(2, 3, 2)`

| 픽셀 (h, w) | (0, 0) | (0, 1) | (0, 2) | (1, 0) | (1, 1) | (1, 2) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| (채널 0, 채널 1) | (0, 1) | (2, 3) | (4, 5) | (6, 7) | (8, 9) | (10, 11) |

- `permute(2, 0, 1)`: 새 0번 축 = 옛 2번 축(C), 새 1번 축 = 옛 0번 축(H), 새 2번 축 = 옛 1번 축(W) → **값은 그대로, 인덱스 (h, w, c)의 값을 [c, h, w] 자리로 옮김**

| 옛 자리 (h, w, c) | 값 | 새 자리 [c, h, w] |
|:---:|:---:|:---:|
| (0, 1, 0) | 2 | [0, 0, 1] |
| (1, 0, 1) | 7 | [1, 1, 0] |

- `reshape`: 값을 한 줄로 펴서 **행 우선 순서로 다시 자름** → 모양만 맞춰 주고 픽셀과 채널을 섞음
- `transpose(0, 2)`: 0번 축(H)과 2번 축(C)만 맞바꿈 → [C, W, H]가 되어 높이와 너비가 뒤바뀜

🔍[비교] 세 함수의 `[0]` 결과

| 방법 | 모양 | `[0]`의 내용 | 채널 0 판 |
|:---:|:---:|:---:|:---:|
| `permute(2, 0, 1)` | [2, 2, 3] | [[0, 2, 4], [6, 8, 10]] | **O** |
| `reshape(2, 2, 3)` | [2, 2, 3] | [[0, 1, 2], [3, 4, 5]] | X (픽셀과 채널이 섞임) |
| `transpose(0, 2)` | [2, 3, 2] | [[0, 6], [2, 8], [4, 10]] | X (높이·너비가 뒤바뀜) |

## 2. 값 범위: 정규화(÷255)

- 이 수업에서 정규화 = 픽셀 값을 255로 나눠 0~1로 맞추는 일
- `astype(np.float32)`는 자료형만 바꾸고 값은 그대로(200 → 200.0) → 범위는 따로 나눠야 함

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 정규화 | 정규화 값 x′ = 픽셀 값 x ÷ 255 | $x' = \dfrac{x}{255}$ |

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 기준 픽셀 (R, G, B) | (200, 100, 50) | |
| 빨강 | 200 ÷ 255 | 0.7843 |
| 초록 | 100 ÷ 255 | 0.3922 |
| 파랑 | 50 ÷ 255 | 0.1961 |

- torchvision v2의 `ToImage()`: 축 순서 HWC → **CHW** (dtype은 그대로 uint8)
- `ToDtype(torch.float32, scale=True)`: float32로 바꾸면서 **÷255** / `scale` 기본값은 `False`라서 빼먹으면 값이 float32가 되기만 하고 범위는 0~255 그대로
- `Conv2d`는 [N, C, H, W]를 받으므로 사진 한 장 앞에 `[None]`으로 크기 1짜리 축을 붙임

## 3. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cv_practice.ipynb)

```python
import torch                                              # PyTorch

x = torch.arange(12).reshape(2, 3, 2)                     # 0~11을 모양 (H=2, W=3, C=2)로 배치
print(x.permute(2, 0, 1)[0])                              # 인덱스를 옮겨 채널 0 판을 꺼냄
print(x.reshape(2, 2, 3)[0])                              # 한 줄로 펴서 다시 자름
print(x.transpose(0, 2)[0])                               # 0번 축과 2번 축만 교환
```

```
tensor([[ 0,  2,  4],
        [ 6,  8, 10]])
tensor([[0, 1, 2],
        [3, 4, 5]])
tensor([[ 0,  6],
        [ 2,  8],
        [ 4, 10]])
```

```python
import numpy as np                                        # 배열 라이브러리
from torchvision.transforms import v2                     # 이미지 변환 모음

img = np.array([[[200, 100, 50], [10, 20, 30]],
                [[0, 255, 128], [90, 90, 90]]], dtype=np.uint8)   # 모양 (H=2, W=2, C=3)
t_img = v2.ToImage()(img)                                 # 변환 도구를 만들고 img에 적용: HWC → CHW
print(tuple(t_img.shape), t_img.dtype)                    # 축 순서만 바뀌고 uint8 유지
scaled = v2.ToDtype(torch.float32, scale=True)(t_img)     # float32로 바꾸고 ÷255
print(scaled[:, 0, 0])                                    # 모든 채널, 0번 행 0번 열 픽셀
plain = v2.ToDtype(torch.float32)(t_img)                  # scale 기본값 False: 값 범위 그대로
print(plain[:, 0, 0])
print(scaled.min().item(), scaled.max().item(), scaled.dtype)   # .item(): 숫자 하나를 파이썬 숫자로 꺼냄
batch = scaled[None]                                      # 맨 앞에 크기 1짜리 축 추가
print(batch.shape)
```

```
(3, 2, 2) torch.uint8
tensor([0.7843, 0.3922, 0.1961])
tensor([200., 100.,  50.])
0.0 1.0 torch.float32
torch.Size([1, 3, 2, 2])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
