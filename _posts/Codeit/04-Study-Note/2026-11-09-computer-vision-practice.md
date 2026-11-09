---
title: "이미지와 컴퓨터비전 - PRACTICE 풀이"
date: 2026-11-09 18:00:00 +0900
categories: [Codeit, '04-Study-Note']
tags: [codeit, 컴퓨터비전, opencv, 데이터증강, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---
### 1. HWC 배열 → CHW 텐서

- 사진을 파이썬으로 읽으면 넘파이 배열 `[H, W, C]` (세로, 가로, 채널) 순서이다.
- PyTorch는 `[C, H, W]` 순서를 기대한다.
- 순서를 바꿀 때는 `permute`를 쓴다.
- `permute(2, 0, 1)`: 새 0번 축은 옛 2번(C), 새 1번 축은 옛 0번(H), 새 2번 축은 옛 1번(W).
- `transpose`는 축 **두 개만** 맞바꾸므로 축 세 개는 두 번 써야 한다.
- `reshape`는 축을 바꾸는 것이 아니라 값을 한 줄로 펼친 뒤 다시 자르는 것이다. shape은 맞아 보여도 색이 섞이고, 오류도 눈에 띄지 않는다.

- 정답
```python
return t.permute(2, 0, 1)
```

- 내가 쓴 답
```python
t = t.permute(2, 0, 1)
return t
```

- 같은 결과: `torch.from_numpy(np.transpose(arr, (2, 0, 1)))`, `v2.ToImage()(arr)`
- 여러 정답을 알아야 다른 사람의 코드를 읽을 수 있다.

### 2. 소금후추 잡음 제거

- 소금후추 잡음: 일부 픽셀이 검은 점(0)이나 흰 점(255)으로 바뀐 잡음이다.
- 박스 필터: 창 안의 값을 같은 비중으로 평균낸다.
- 가우시안 필터: 가운데에 큰 비중을 두고 평균낸다.
- 두 필터는 잡음 값이 평균에 들어가서 주변으로 퍼진 **흐린 얼룩**을 남긴다.
- 중앙값 필터: 창 안의 값을 정렬해서 가운데 값을 **고른다.** 잡음(0, 255)은 정렬하면 양 끝으로 가서 결과에 나오지 않는다.

📍[예시] 3×3 창 `52 50 49 / 51 255 53 / 50 48 52`

| 필터 | 계산 | 결과 |
|:---:|:---:|:---:|
| 박스 | 660 ÷ 9 | **73.3** |
| 중앙값 | 정렬 후 5번째 | **51** |

- 정답

```python
return cv2.medianBlur(noisy, 3) 
```
- 내가 쓴 답

```python
filtered = cv2.medianBlur(noisy, 3)
return filtered
```

### 3. 밝아지는 경계와 어두워지는 경계 동시 검출

- 경계: 이웃한 픽셀의 밝기가 크게 바뀌는 곳이다.
- 밝아지는 경계(어두움 → 밝음)는 Sobel 결과가 **양수**, 어두워지는 경계(밝음 → 어두움)는 **음수**이다.
- 두 경계를 모두 보려면 음수를 담을 수 있는 자료형에 받고 **절댓값**으로 세기만 남긴다.
- uint8은 0~255만 담을 수 있어서 음수가 0으로 잘리고, 어두워지는 경계를 놓친다.
- 그래서 `int16` 이상(`CV_16S`, `CV_64F`)으로 받는다.

📍[예시] x 방향 Sobel, 각 행이 같은 3×3 창

| 경우 | 창 | 결과 |
|:---:|:---:|:---:|
| 밝아지는 경계 | 20 20 60 | **+160** |
| 어두워지는 경계 | 60 60 20 | **−160** |

- 정답

```python
gx = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
   return np.abs(gx)
```

- 내가 쓴 답

```python
sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = np.abs(sobel_x)
return sobel_x
```

🔍[비교] 정답은 `cv2.CV_16S`(16비트 정수)를 쓰고 내 풀이는 `cv2.CV_64F`(실수)를 썼다. 둘 다 음수를 담을 수 있어서 정답이다.

- `cv2.convertScaleAbs(gx)`는 255에서 잘려 두 세기가 모두 255가 되지만, 두 값이 같으므로 검증은 통과한다.

### 4. 좌우 반전 바운딩 박스 변환

- 입력은 XYXY 박스 `(x1, y1, x2, y2)`이다. `(x1, y1)`은 왼쪽 위, `(x2, y2)`는 오른쪽 아래 모서리이다.
- XYXY의 약속: 항상 `x1 < x2`, `y1 < y2`이다.
- 좌우 반전하면 $x' = W - x$ 이다. ($W$: 이미지 너비)
- 그대로 $W - x_1$, $W - x_2$를 쓰면 순서가 뒤집혀 `x1 > x2`가 된다.
- 그래서 **새 x1 = W − 옛 x2, 새 x2 = W − 옛 x1**로 자리를 바꿔 준다. y는 그대로이다.

- 정답

```python
x1, y1, x2, y2 = box
   return (width - x2, y1, width - x1, y2)
```

- 내가 쓴 답

```python
new_x1 = width - x2
new_x2 = width - x1
return (new_x1, y1, new_x2, y2)
```

### 5. 데이터 증강과 평가 경로

- 데이터 증강은 **학습할 때만** 한다. 과적합(학습 데이터에만 지나치게 맞춰지는 현상)을 완화하고 성능을 높인다.
- 평가에 증강을 쓰면 같은 이미지도 결과가 달라져서 제대로 된 평가를 할 수 없다. 평가 경로에 실수로 넣지 않았는지 확인한다.
- 평가 경로는 무작위 변환만 빼고, 나머지 처리는 학습과 똑같이 맞춘다.

🔍[비교] 학습 경로와 평가 경로

| 단계 | 학습 | 평가 |
|:---:|:---:|:---:|
| 이미지 타입 변환 (HWC → CHW) | ToImage | ToImage |
| 크기 맞추기 | **RandomResizedCrop** (무작위) | **Resize + CenterCrop** (고정) |
| 좌우 뒤집기 | **RandomHorizontalFlip** (무작위) | **없음** |
| ÷255 | ToDtype(scale=True) | ToDtype(scale=True) |
| 표준화 | Normalize(MEAN, STD) | Normalize(MEAN, STD) |

- `RandomResizedCrop`의 `scale=(0.5, 1.0)`: 원본 면적의 50~100% 중 무작위 비율로 잘라낸 뒤 목표 크기로 맞춘다.
- `Resize`는 짧은 변을 목표 크기로 맞추고, `CenterCrop`은 가운데를 목표 크기로 잘라 크기를 고정한다.
- 뒤집기 확률은 `p`로 바꾼다. 기본값은 0.5이다. (RandomHorizontalFlip(p=0.3) → 30% 확률로 좌우 뒤집기)
- `ToDtype(torch.float32, scale=True)`는 float32로 바꾸면서 값을 ÷255 하여 0~1 범위로 맞춘다.
- `Normalize`(표준화)는 0~1 범위를 전제로 하므로 ÷255가 먼저 온다. 두 단계는 별개이다.
- 학습과 평가의 입력 크기, MEAN, STD는 같아야 한다.

- 정답

```python
   v2.ToImage(),
   v2.Resize(160),
   v2.CenterCrop(160),
   v2.ToDtype(torch.float32, scale=True),
   v2.Normalize(MEAN, STD),
])
```
- 내가 쓴 답

```python
   v2.ToImage(),
   v2.Resize(160),
   v2.CenterCrop(160),
   v2.ToDtype(torch.float32, scale=True),
   v2.Normalize(MEAN, STD),
])
```

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 공부한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
