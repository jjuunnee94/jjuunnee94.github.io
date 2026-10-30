---
title: "PyTorch Basic - 02. 행렬곱과 모양"
date: 2026-10-27 12:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, matmul, broadcasting, view, reshape, squeeze, unsqueeze, cat, stack, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
---
 
### 1. 행렬곱 `@`

- 행렬곱은 **줄과 칸을 짝지어 곱한 뒤 전부 더하는** 계산.

```python
import torch

X = torch.tensor([[ 80., 90.],     # 학생 0: 국어, 수학
                  [ 70., 60.],     # 학생 1
                  [100., 50.]])    # 학생 2
W = torch.tensor([[0.25],          # 국어 반영 비율
                  [0.75]])         # 수학 반영 비율

print(X.shape, W.shape)   # torch.Size([3, 2]) torch.Size([2, 1])
print(X @ W)
print((X @ W).shape)
```

```
torch.Size([3, 2]) torch.Size([2, 1])
tensor([[87.5000],
        [62.5000],
        [62.5000]])
torch.Size([3, 1])
```

- 첫 번째 학생: `80 × 0.25 + 90 × 0.75 = 20 + 67.5 = 87.5`
- 수식으로 쓰면 `결과[i] = (X의 i번 줄의 각 칸 × W의 각 값)의 합` 이다.
- 즉 **줄 하나(특성들)와 W 한 칸 묶음을 짝지어 곱하고 합한 값**이 결과 한 칸이 된다.

**모양 규칙**

```
(N, D) @ (D, K)  →  (N, K)
```

- 가운데 `D`(특성 개수)가 **서로 같아야** 한다.
- 바깥의 `N`(표본 수)과 `K`(출력 개수)가 결과 모양이 된다.
- W의 모양은 `(특성 수, 출력 수)` 로 기억하면 된다.

**`*`와의 차이**

- `*`는 같은 자리끼리만 곱하고 **더하지 않는다.** 곱한 뒤 더하면 `@`와 같은 값이 나온다.

```python
w = torch.tensor([0.25, 0.75])
print(X * w)
print((X * w).sum(dim=1))
```

```
tensor([[20.0000, 67.5000],
        [17.5000, 45.0000],
        [25.0000, 37.5000]])
tensor([87.5000, 62.5000, 62.5000])
```

**출력이 여러 개일 때 (K > 1)**

```python
W2 = torch.tensor([[0.25, 0.5],
                   [0.75, 0.5]])    # (2, 2): 점수 방식 2가지
print(X @ W2)
print((X @ W2).shape)
```

```
tensor([[87.5000, 85.0000],
        [62.5000, 65.0000],
        [62.5000, 75.0000]])
torch.Size([3, 2])
```

- 오른쪽 행렬의 칸(열) 하나가 출력 하나를 만든다.
- 가운데 숫자가 다르면 에러가 난다.

```python
X @ torch.ones(3, 1)    # (3, 2) @ (3, 1)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x2 and 3x1)
```

### 2. 브로드캐스팅 (Broadcasting)

- 모양이 다른 Tensor끼리 계산할 때, 길이가 1인 방향을 **자동으로 늘려서** 맞춰 주는 규칙이다.
- 규칙은 세 가지다.
    - 두 shape를 **오른쪽 끝부터** 맞춰 놓는다.
    - 자리마다 두 숫자가 **같거나, 둘 중 하나가 1**이어야 한다.
    - 길이가 모자란 쪽은 **왼쪽에** 1이 있다고 본다.
- 한 자리라도 어긋나면 에러가 난다.

**예시 ① 줄마다 같은 값 더하기**

```python
b = torch.tensor([10., 20.])      # (2,)
print(X + b)
```

```
tensor([[ 90., 110.],
        [ 80.,  80.],
        [110.,  70.]])
```

- `(3, 2)`와 `(2,)`를 오른쪽부터 맞추면 `2 = 2`, 왼쪽은 비어 있으므로 `(1, 2)`로 보고 `3`줄로 늘린다.
- 결과 shape는 `(3, 2)`이다.

**예시 ② 칸 방향과 줄 방향이 만나는 경우**

```python
print((torch.zeros(4, 1) + torch.zeros(1, 5)).shape)   # torch.Size([4, 5])
```

- `(4, 1)`과 `(1, 5)`는 자리마다 한쪽이 1이라서, 서로 늘어나 `(4, 5)`가 된다.

**예시 ③ 맞지 않는 경우**

```python
torch.zeros(2, 3) + torch.zeros(3, 1)
# RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 0
```

- 오른쪽 끝 `3`과 `1`은 괜찮지만, 그다음 `2`와 `3`은 둘 다 1이 아니고 서로 다르다.

**예시 ④ 에러 없이 틀리는 위험한 경우**

```python
col = torch.tensor([[1], [2], [3]])     # (3, 1)
row = torch.tensor([10, 20, 30])        # (3,)

print((col - row).shape)                # torch.Size([3, 3])
print(col - row)
```

```
torch.Size([3, 3])
tensor([[ -9, -19, -29],
        [ -8, -18, -28],
        [ -7, -17, -27]])
```

- `(3, 1)`과 `(3,)`는 `(3, 1)`과 `(1, 3)`으로 해석되어 `(3, 3)`이 된다. 에러 없이 9개가 나온다.
- 예측값 `(N, 1)`에서 정답 `(N,)`을 빼는 실수가 이 경우다. 손실 값이 조용히 틀어진다.
- 해결은 모양을 직접 맞추는 것이다.

```python
print(col - row.unsqueeze(1))           # (3, 1) - (3, 1)
```

```
tensor([[ -9],
        [-18],
        [-27]])
```

### 3. 모양 바꾸기: `view`와 `reshape`

- 원소의 **순서와 개수는 그대로** 두고 묶는 방식(shape)만 바꾼다.
- 바꾸기 전과 후의 원소 개수가 같아야 한다.

```python
v = torch.arange(12)
print(v.view(3, 4))
print(v.view(2, -1).shape)    # torch.Size([2, 6])
print(v.view(-1, 6).shape)    # torch.Size([2, 6])
```

```
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
torch.Size([2, 6])
torch.Size([2, 6])
```

- `-1`은 "알아서 계산해 줘"라는 뜻이다. `12 ÷ 2 = 6`으로 맞춰진다.
- `-1`은 **한 번만** 쓸 수 있다.
- 원소 개수가 맞지 않으면 에러가 난다.

```python
v.view(5, -1)
# RuntimeError: shape '[5, -1]' is invalid for input of size 12
```

**`view`는 같은 데이터를 공유한다**

```python
a = v.view(3, 4)
a[0, 0] = 99
print(v[0])      # tensor(99)
```

- `view`는 데이터를 복사하지 않고 **읽는 방식만 바꾼다.** 한쪽을 고치면 다른 쪽도 바뀐다.

**`view`가 안 될 때는 `reshape`**

```python
t = torch.arange(6).view(2, 3).T       # (3, 2)
print(t)
```

```
tensor([[0, 3],
        [1, 4],
        [2, 5]])
```

```python
t.view(6)
# RuntimeError: view size is not compatible with input tensor's size and stride ...
print(t.reshape(6))                    # tensor([0, 3, 1, 4, 2, 5])
```

- .T (전치)는 데이터 위치를 실제로 이동시키지 않고 '읽는 순서만' 뒤집습니다.
- 따라서 컴퓨터 메모리에는 여전히 원래 순서 (0, 1, 2, 3, 4, 5)대로 연속하게 들어있기 때문에, 메모리가 연속적이어야 하는 view()는 동작하지 않고 에러가 납니다.
- reshape(): 어떤 상황에서도 안전하게 동작하지만, 경우에 따라 복사본(새 메모리)이 만들어질 수 있습니다.

### 4. `squeeze`와 `unsqueeze`

- `squeeze(dim)`: 길이가 1인 방향을 **없앤다.**
- `unsqueeze(dim)`: 길이 1짜리 방향을 **끼워 넣는다.**

```python
a = torch.tensor([[7, 8, 9, 10]])           # (1, 4)
b = torch.tensor([[7], [8], [9], [10]])     # (4, 1)

print(a.squeeze(0).shape)      # torch.Size([4])
print(b.squeeze(1).shape)      # torch.Size([4])
print(a.squeeze().shape, b.squeeze().shape)   # torch.Size([4]) torch.Size([4])
```

- `squeeze()`에 번호를 안 주면 길이 1인 방향을 **전부** 없앤다.
- `(1, 4)`와 `(4, 1)`은 서로 다른 모양인데, 없애고 나면 둘 다 `(4,)`가 되어 **어느 쪽에 방향이 있었는지 정보가 사라진다.**
- 되돌릴 때는 직접 정해 줘야 한다.

```python
v = a.squeeze()
print(v.unsqueeze(0).shape)    # torch.Size([1, 4])
print(v.unsqueeze(1).shape)    # torch.Size([4, 1])
```

- 길이가 1이 아닌 방향에 `squeeze(번호)`를 쓰면 아무 일도 일어나지 않는다.

### 5. 붙이기: `cat`과 `stack`

- `cat`은 **있는 방향에 이어 붙이고**, `stack`은 **새 방향을 만들어 쌓는다.**
- **dim (방향 번호)**: `shape`에서 왼쪽부터 `0`부터 세며, 2차원 표(`shape (행, 열)`) 기준은 다음과 같습니다.
  - **`dim=0`**: **행(줄/위아래) 방향**으로 붙임 ➔ 줄 수가 늘어남 (아래로 연결)
  - **`dim=1`**: **열(칸/옆) 방향**으로 붙임 ➔ 칸 수가 늘어남 (옆으로 연결)

```python
p = torch.tensor([[1, 2, 3]])    # (1, 3)
q = torch.tensor([[4, 5, 6]])    # (1, 3)

print(torch.cat([p, q], dim=0))
print(torch.cat([p, q], dim=1))
```

```
tensor([[1, 2, 3],
        [4, 5, 6]])
tensor([[1, 2, 3, 4, 5, 6]])
```

- `cat`은 고른 방향의 길이만 더해진다. `dim=0`은 `(2, 3)`, `dim=1`은 `(1, 6)`이다.
- 나머지 방향의 길이는 서로 같아야 한다.

```python
print(torch.stack([p, q], dim=0))
print(torch.stack([p, q], dim=0).shape)
print(torch.stack([p, q], dim=1).shape)
print(torch.stack([p, q], dim=2).shape)
```

```
tensor([[[1, 2, 3]],

        [[4, 5, 6]]])
torch.Size([2, 1, 3])
torch.Size([1, 2, 3])
torch.Size([1, 3, 2])
```

- `stack`은 `p`, `q`를 그대로 두고 **새 방향(길이 = 합칠 개수)** 을 `dim` 자리에 끼운다.
- 3차원 Tensor는 `(덩어리, 줄, 칸)`으로 읽는다. 숫자를 뒤에서부터 칸 → 줄 → 덩어리로 묶으면 된다.

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }

