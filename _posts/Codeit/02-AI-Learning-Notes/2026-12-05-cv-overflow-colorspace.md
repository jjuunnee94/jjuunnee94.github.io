---
title: "컴퓨터 비전 - 02. uint8 오버플로와 색공간"
date: 2026-12-05 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, cv, computer-vision, uint8, color-space, hsv, opencv, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. uint8 오버플로

- uint8은 0~255만 담음 → 범위를 넘으면 **오류 없이** 256으로 나눈 나머지가 남음
- 빼기도 같음: `np.array([5], dtype=np.uint8) - 10`은 −5가 아니라 **251**

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 오버플로 | 결과 = (a + b)를 256으로 나눈 나머지 | $(a + b) \bmod 256$ |

📍[예시] 픽셀 R = 200에 100을 더하면 밝아지지 않고 어두워진다.

| 단계 | 계산 | 결과 |
|:---:|:---:|:---:|
| 원래 값 | 픽셀 R = 200 | |
| 더하기 | 200 + 100 | 300 (255 초과) |
| 256으로 나눈 나머지 | 300 − 256 | **44** |

- 바른 방법: **넓은 dtype(`int16`)으로 계산한 뒤 `np.clip`으로 자르고 `uint8`으로 되돌림**
- `astype(자료형)`: 값은 그대로, 자료형만 바꿈 / `np.clip(배열, 최솟값, 최댓값)`: 범위 밖 값을 경계 값으로 바꿈

| 방법 | 200 칸 | 30 칸 |
|:---:|:---:|:---:|
| uint8 그대로 `+ 100` | 44 | 130 |
| int16으로 계산 | 300 | 130 |
| `np.clip(…, 0, 255)` 후 uint8 | **255** | **130** |

## 2. 색공간

- 색공간(color space): 색을 세 수로 나타내는 좌표 방식
- RGB는 색과 밝기가 한 숫자에 섞여 있어서, 같은 주황도 밝기에 따라 값이 크게 다름

| 픽셀 | 빨강(R) | 초록(G) | 파랑(B) | 느낌 |
|:---:|:---:|:---:|:---:|:---:|
| 밝은 주황 | 200 | 100 | 50 | 주황 |
| 어두운 주황 | 100 | 50 | 25 | 같은 주황, 어두움 |

- HSV: 색을 **색상(H)·채도(S)·명도(V)** 세 축으로 나타냄

| 기호 | 이름 | 뜻 | OpenCV 8비트 범위 |
|:---:|:---:|:---:|:---:|
| 색상(H) | 어떤 색인가 | 색의 종류 | 0~179 |
| 채도(S) | 얼마나 선명한가 (0이면 회색) | 색의 진한 정도 | 0~255 |
| 명도(V) | 얼마나 밝은가 | 가장 큰 채널 값 | 0~255 |

- 색상(H)은 원래 0~360도인데 OpenCV는 8비트에 담으려고 절반(0~179)으로 씀

| 픽셀 (R, G, B) | 색상(H) | 채도(S) | 명도(V) | 설명 |
|:---:|:---:|:---:|:---:|:---:|
| (200, 100, 50) | 10 | 191 | 200 | 밝은 주황 |
| (100, 50, 25) | 10 | 191 | **100** | 같은 주황, **명도만 절반** |
| (30, 90, 200) | 109 | 217 | 200 | 파랑 |
| (60, 60, 60) | 0 | **0** | 60 | 회색 (채도 0) |

- 두 주황은 색상(H)과 채도(S)가 같고 명도(V)만 다름 → "색상(H) 5~20, 채도(S) 100 초과" 조건 하나로 둘 다 선택
- HSV는 계산이 단순한 대신, 같은 색상 간격이라도 사람이 느끼는 차이는 색 영역마다 다름

| 색공간 | 축 | 쓰는 곳 |
|:---:|:---:|:---:|
| RGB | 빨강, 초록, 파랑 | 화면 표시, 파일 |
| HSV | 색상, 채도, 명도 | 색 범위로 영역 선택 |
| YCbCr | 밝기 Y, 색차 Cb·Cr | JPEG 압축 |

## 3. BGR과 RGB

- `cv2.imread`로 읽은 사진의 채널 순서는 (B, G, R) → 그대로 그리면 오류 없이 색이 뒤바뀜

| 단계 | 픽셀 | 뜻 |
|:---:|:---:|:---:|
| 원래 RGB | [200, 100, 50] | 주황 |
| OpenCV가 읽은 BGR | [50, 100, 200] | 같은 숫자를 RGB로 그리면 파랑 |
| `cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)` | [200, 100, 50] | 다시 주황 |

## 4. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cv_practice.ipynb)

```python
import numpy as np                                      # 배열 라이브러리
import cv2                                              # OpenCV: 이미지 처리 라이브러리

px = np.array([200, 30], dtype=np.uint8)                # uint8 픽셀 값 2개
print(px + 100)                                         # uint8 그대로 더하면 256으로 돌아감
wide = px.astype(np.int16) + 100                        # 넓은 dtype(int16)으로 바꿔서 계산
safe = np.clip(wide, 0, 255).astype(np.uint8)           # 0~255로 자른 뒤 uint8으로 되돌림
print(safe)                                             # 255에서 멈춤
```

```
[ 44 130]
[255 130]
```

```python
rgb = np.array([[[200, 100, 50], [100, 50, 25]],
                [[30, 90, 200], [60, 60, 60]]], dtype=np.uint8)   # 픽셀 4개, 모양 (2, 2, 3)
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)              # RGB → HSV, 채널이 (H, S, V)가 됨
print(hsv[:, :, 0])                                     # 모든 행·열의 0번 채널 = 색상(H), 모양 (2, 2)
sel = (hsv[:, :, 0] >= 5) & (hsv[:, :, 0] <= 20) & (hsv[:, :, 1] > 100)   # 조건 3개를 모두 만족(AND)
print(sel)                                              # 조건에 맞는 칸이 True인 bool 배열
print(sel.sum())                                        # True의 개수
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)              # R과 B 순서를 바꿈(OpenCV가 읽은 모양)
print(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[0, 0])       # 다시 RGB로: 0번 행 0번 열 픽셀
```

```
[[ 10  10]
 [109   0]]
[[ True  True]
 [False False]]
2
[200 100  50]
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
