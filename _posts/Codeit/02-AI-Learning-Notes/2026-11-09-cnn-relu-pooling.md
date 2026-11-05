---
title: "CNN - ReLU, Pooling, 수용영역"
date: 2026-11-09 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, cnn, relu, maxpool, pooling, receptive-field, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

### 1. ReLU : 음수는 0, 양수는 그대로

- $\text{ReLU}(x) = \max(0,\ x)$
- **각 칸마다 따로** 적용하고, 채널 수와 공간 크기는 바뀌지 않는다.
- 합성곱은 "곱해서 더하기"라서 그대로 쌓으면 하나의 합성곱과 같아진다. 중간에 ReLU 같은 **비선형** 함수를 끼워야 층을 쌓는 의미가 생긴다.

📍 [예시]

| 적용 전 | 적용 후 |
| :---: | :---: |
| 3 &nbsp; -2 &nbsp; 0<br>-1 &nbsp; 4 &nbsp; -5 | 3 &nbsp; 0 &nbsp; 0<br>0 &nbsp; 4 &nbsp; 0 |

- 음수의 크기 정보는 사라지지만, 남은 값의 **위치**는 그대로이다.

### 2. Max Pooling : 작은 영역을 대표값 하나로

- 입력을 겹치지 않는 2×2 창으로 나누고, 창마다 **최댓값 하나**만 남긴다. (`MaxPool2d(2)`의 stride는 기본으로 창 크기와 같은 2)
- 학습하는 가중치가 **없다.** 합이나 평균이 아니라 창 안에 **실제로 있는** 최댓값이다.
- 값이 같은 창은 안의 배치가 달라져도 최댓값은 그대로이다.

![4×4 입력에서 2×2 창마다 최댓값을 남기는 Max Pooling](/assets/02-AI-Learning-Notes-paper/11-09-cnn_maxpool.png)

📍 [예시]

| 창 (시작 위치) | 네 값 | 최댓값 |
| :---: | :---: | :---: |
| (0,0) | 7, 2, 3, 6 | **7** |
| (0,2) | 5, 1, 0, 4 | **5** |
| (2,0) | 1, 8, 9, 0 | **9** |
| (2,2) | 2, 3, 6, 5 | **6** |

- 출력은 `[[7, 5], [9, 6]]`이다.
- 크기는 출력 크기 공식에 $k=2,\ s=2,\ p=0$을 넣으면 된다. 짝수 $L$이면 $L/2$이다. (28 → 14 → 7)

$$
L_{out} = \left\lfloor \frac{L - 2}{2} \right\rfloor + 1
$$

### 3. 수용영역 : 쌓을수록 넓게 본다

- **수용영역(Receptive Field)** 은 출력 한 칸이 원본 입력에서 **영향을 받을 수 있는 범위**이다.
- 2층 출력 한 칸은 1층 출력 3×3(9칸)을 읽는다. 이 9칸은 이웃이라 서로 많이 **겹치므로**, 원본에서는 9×9가 아니라 합쳐서 5×5가 된다.

![3×3 합성곱을 쌓으면 출력 한 칸이 읽는 입력 범위가 3, 5, 7로 넓어지는 한 줄 그림](/assets/02-AI-Learning-Notes-paper/11-09-cnn_receptive.png)

- 3×3 합성곱(stride 1)을 $n$층 쌓으면 한 층마다 각 축으로 $k-1=2$씩 넓어진다.

$$
RF_n = 1 + n\,(k-1)
$$

📍 [예시] 입력 9×9, 3×3 합성곱 4층 (stride 1, padding 0)

| 층 | 출력 공간 크기 | 원본 입력의 수용영역 |
| :---: | :---: | :---: |
| **Conv 1** | 7×7 | 3×3 |
| **Conv 2** | 5×5 | 5×5 |
| **Conv 3** | 3×3 | 7×7 |
| **Conv 4** | 1×1 | 9×9 |

⚠️ [주의] 수용영역은 영향을 줄 수 있는 **범위**일 뿐, 모든 위치가 같은 세기로 영향을 준다는 뜻은 아니다. stride가 2 이상이거나 Pooling이 끼면 더 빠르게 넓어진다.

### 4. 💻 [코드]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cnn_practice.ipynb)

```python
import torch
from torch import nn
import torch.nn.functional as F

a = torch.tensor([[ 3., -2.,  0.],
                  [-1.,  4., -5.]]).reshape(1, 1, 2, 3)  # [N, C, H, W]
print(F.relu(a)[0, 0])  # relu는 가중치가 없어서 함수(F)로 바로 사용 / [0, 0] : 첫 이미지의 첫 채널

y = torch.tensor([[7., 2., 5., 1.],
                  [3., 6., 0., 4.],
                  [1., 8., 2., 3.],
                  [9., 0., 6., 5.]]).reshape(1, 1, 4, 4)
pool = nn.MaxPool2d(kernel_size=2)  # stride를 안 쓰면 창 크기(2)와 같아서 창이 겹치지 않음
print(pool(y).shape)
print(pool(y)[0, 0])
print(sum(p.numel() for p in pool.parameters()))  # 학습 가능한 값의 개수 → 0
```

```
tensor([[3., 0., 0.],
        [0., 4., 0.]])
torch.Size([1, 1, 2, 2])
tensor([[7., 5.],
        [9., 6.]])
0
```

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
