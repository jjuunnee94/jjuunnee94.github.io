---
title: "CNN - 합성곱 계산과 특징맵"
date: 2026-11-05 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, cnn, convolution, filter, feature-map, conv2d, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

### 1. 합성곱

- 출력 한 칸은 두 단계로 만든다.
  - 입력의 3×3 **패치(patch)** 와 3×3 **필터(filter)** 를 **같은 위치끼리 곱한다.**
  - 곱한 9개를 **전부 더한다.**
- 이것을 **원소별 곱의 합**이라고 한다. 행렬곱이 아니라 같은 좌표끼리만 곱한다.

📍 [예시]

| 입력 패치 | 필터 | 원소별 곱 |
| :---: | :---: | :---: |
| 3 &nbsp; 1 &nbsp; 0<br>2 &nbsp; 4 &nbsp; 1<br>0 &nbsp; 2 &nbsp; 1 | 1 &nbsp; 1 &nbsp; 1<br>0 &nbsp; 0 &nbsp; 0<br>-1 &nbsp; -1 &nbsp; -1 | 3 &nbsp; 1 &nbsp; 0<br>0 &nbsp; 0 &nbsp; 0<br>0 &nbsp; -2 &nbsp; -1 |

| 줄 | 계산 | 결과 |
| --- | --- | --- |
| 윗줄 | 3×1 + 1×1 + 0×1 | 4 |
| 가운데 줄 | 2×0 + 4×0 + 1×0 | 0 |
| 아랫줄 | 0×(-1) + 2×(-1) + 1×(-1) | -3 |

-  `4 + 0 + (-3) = 1`

### 2. 특징맵

- 한 칸을 구한 뒤 필터를 한 칸씩 옮기며 같은 계산을 반복한다. **필터 값은 그대로이고, 필터 아래에 놓이는 입력 패치만 바뀐다.** 이것이 가중치 공유이다.

📍 [예시]

| 입력 (4×4) |  필터 |
| :---: | :---: |
| 1 &nbsp; 2 &nbsp; 0 &nbsp; 1<br>3 &nbsp; 1 &nbsp; 1 &nbsp; 2<br>2 &nbsp; 0 &nbsp; 4 &nbsp; 0<br>1 &nbsp; 3 &nbsp; 2 &nbsp; 1 | 1 &nbsp; 1 &nbsp; 1<br>0 &nbsp; 0 &nbsp; 0<br>-1 &nbsp; -1 &nbsp; -1 |

- 각 위치별 계산값
  
| 특징맵 (2×2) |
| :---: |
| -3 &nbsp; -1<br>-1 &nbsp; -2 |

### 3. 코드

📌 [참고] `nn.Conv2d`의 필터는 원래 학습으로 정하는 값이지만, 계산 원리를 보려고 직접 정한 값을 넣는다.

```python
import torch
from torch import nn

x = torch.tensor([[1., 2., 0., 1.],
                  [3., 1., 1., 2.],
                  [2., 0., 4., 0.],
                  [1., 3., 2., 1.]])  # 숫자 뒤에 .을 붙여 float로 만듦 (Conv2d는 정수 Tensor를 받으면 타입 오류)

x = x.reshape(1, 1, 4, 4)
# [N, C, H, W] → (N)이미지 1장, (C)채널 1개(흑백), (H)높이 4, (W)너비 4

K = torch.tensor([[ 1.,  1.,  1.],
                  [ 0.,  0.,  0.],
                  [-1., -1., -1.]])  # 직접 정한 3×3 필터

conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
# (입력 채널 1, 출력 채널(필터 개수) 1, 필터 크기 3 x 3, 편향 사용 X) → 필터 값은 처음엔 랜덤

print(conv.weight.shape)  

with torch.no_grad():
    conv.weight.copy_(K.reshape(1, 1, 3, 3))  # copy_ : 끝에 _가 붙은 함수는 원본을 직접 바꿈

out = conv(x)
print(out.shape)
print(out[0, 0].detach())  # 앞의 두자리 빼고 뒤의 특징맵만 호출
```

```
torch.Size([1, 1, 3, 3]) # 가중치 : [출력 채널, 입력 채널, 필터 높이, 필터 너비]
torch.Size([1, 1, 2, 2]) # 특징맵 : [이미지 개수(Batch), 출력 채널, 특징맵 높이, 특징맵 너비]
tensor([[-3., -1.],
        [-1., -2.]])
```
&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
