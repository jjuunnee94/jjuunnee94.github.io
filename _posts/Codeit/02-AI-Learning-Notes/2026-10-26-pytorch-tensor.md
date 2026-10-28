---
title: "PyTorch Basic - Tensor"
date: 2026-10-26 16:50:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, tensor, shape, dtype, indexing, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
---

### 1. Tensor란?

**Tensor**는 같은 종류의 숫자를 격자 모양으로 담아 두는 PyTorch의 기본 자료구조입니다.

- 한 **줄** = 데이터 1건 (**표본**)
- 한 **칸** = 데이터 속성 1개 (**특성**)

학생 3명의 키(cm)와 몸무게(kg)를 Tensor로 만들면 다음과 같습니다.

```python
import torch

body = torch.tensor([[170, 65],
                     [160, 50],
                     [180, 80]])   # 표본 3개, 특성 2개
```

### 2. 괄호의 역할

| 괄호 | 역할 | 예 |
| --- | --- | --- |
| `[ ]` | 값을 묶거나, 번호로 고름 | `[1, 2]`, `body[1]` |
| `( )` | 함수 호출, shape 같은 값 묶음(튜플) | `torch.zeros((3, 2))` |
| `{ }` | 이름과 값을 짝지은 사전(dict) | `{'w': 1.5}` |

- `[ ]` **한 겹이 방향 하나**입니다. 겹이 곧 `ndim`입니다.
- `(3,)`의 쉼표는 "원소가 하나인 튜플"이라는 표기입니다. 방향이 하나이고 길이가 3이라는 뜻입니다.

```python
torch.tensor(5)                   # 괄호 없음 → ndim 0, 숫자 하나
torch.tensor([5, 7, 9])           # [ ] 한 겹 → ndim 1, shape (3,)
torch.tensor([[5, 7, 9],
              [1, 2, 3]])         # [[ ]] 두 겹 → ndim 2, shape (2, 3)
```

### 3. Tensor 읽기 : shape, ndim, numel, dtype

| 속성 | 뜻 | `body`의 값 |
| --- | --- | --- |
| `.shape` | 각 방향의 길이 `(줄, 칸)` | `(3, 2)` |
| `.ndim` | 방향(축)의 개수 | `2` |
| `.numel()` | 전체 숫자 개수 (3 × 2) | `6` |
| `.dtype` | 숫자의 종류 | `torch.int64` |

방향이 3개인 예로, 한 학년 3개 반의 성적표(반 3개 × 학생 4명 × 과목 2개)입니다.

```python
scores = torch.zeros((3, 4, 2))
print(scores.shape)     # torch.Size([3, 4, 2])
print(scores.ndim)      # 3
print(scores.numel())   # 24  (3 × 4 × 2)
```

### 4. dtype : 숫자의 종류

| 만드는 방법 | dtype |
| --- | --- |
| 소수점 없는 숫자 (`170`) | `int64` (정수) |
| 소수점 있는 숫자 (`170.0`) | `float32` (실수) |
| `True`, `False` | `bool` (참/거짓) |

- 정수와 실수를 섞으면 전체가 `float32`로 맞춰집니다. (`torch.tensor([[1, 2.5]])` → `float32`)
- 학습용 값은 기울기를 구해야 하므로 **실수(`float32`)** 여야 합니다. 그래서 숫자 뒤에 점(`.`)을 붙여 만듭니다.

```python
body_f = torch.tensor([[170., 65.],
                       [160., 50.],
                       [180., 80.]])
print(body_f.dtype)   # torch.float32
```

### 5. 쉽게 만드는 도구

- `torch.zeros(shape)` / `torch.ones(shape)` : 0 또는 1로 채운 Tensor (기본 `float32`)
- `torch.arange(시작, 끝, 간격)` : 끝 **미만**까지, 간격 기본값 1

```python
print(torch.zeros((2, 3)))        # [[0., 0., 0.], [0., 0., 0.]]
print(torch.ones((2, 3)))         # [[1., 1., 1.], [1., 1., 1.]]
print(torch.arange(5))            # tensor([0, 1, 2, 3, 4])
print(torch.arange(3, 12, 3))     # tensor([3, 6, 9])  12미만까지
print(torch.arange(10, 0, -3))    # tensor([10, 7, 4, 1])   0을 넘기기 전까지
```

### 6. 원소 고르기 (인덱싱·슬라이싱)

- 번호는 **0부터** 센다
- 구간 `시작:끝`은 **시작 이상, 끝 미만** (`시작 ≤ 번호 < 끝`). 가져오는 개수는 `끝 − 시작`
- `:` 만 있을 시 그 방향 **전부**
- 쉼표 앞은 줄(0번 방향), 뒤는 칸(1번 방향)
- **번호 하나**로 고르면 그 방향이 **사라지고**, **구간**으로 고르면 그 방향이 **남는다**

```python
price = torch.tensor([[100, 200, 300],
                      [400, 500, 600]])      # 상품 2종 × 가게 3곳, shape (2, 3)

print(price[1, 2])      # tensor(600)               줄 1번, 칸 2번 → 0번부터 셈, 숫자 하나
print(price[1])         # tensor([400, 500, 600])   shape (3,)   줄 1번 전체
print(price[:, 0])      # tensor([100, 400])        shape (2,)   번호 하나라서 칸 방향 사라짐

print(price[:, 0:2])    # tensor([[100, 200],
                        #         [400, 500]])     shape (2, 2) 0 ≤ 번호 < 2 이므로 0, 1번 칸

print(price[:, 1:3])    # tensor([[200, 300],
                        #         [500, 600]])     shape (2, 2) 3 − 1 = 2칸

print(price[:, 1:2])    # tensor([[200], [500]])    shape (2, 1) 구간이라서 칸 방향 남음
```

방향이 늘어도 규칙은 같습니다. 쉼표로 방향을 나누고, 번호 하나로 고른 방향은 사라집니다.

```python
scores = torch.arange(24).reshape(3, 4, 2)   # 0~23을 (3, 4, 2) 모양으로 (reshape은 데이터 순서 유지하며 모양 변경)
print(scores[0].shape)         # torch.Size([4, 2])   1반 표 전체
print(scores[0, 2].shape)      # torch.Size([2])      1반 3번째 학생
print(scores[0, 2, 1])         # tensor(5)            1반 3번째 학생의 1번 칸
```

번호 하나로 고르면 방향이 사라져서 `(3,)`, `(2,)`가 되고, 구간으로 고르면 방향이 남아서 `(2, 2)`, `(2, 1)`이 됩니다. 정답 데이터를 `(N, 1)` 모양으로 맞춰야 하는 경우가 많아서 이 차이가 이후 학습 코드에서 계속 중요합니다.

<div class="alert alert-info mt-4 mb-4" role="alert">
  이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. 
  (공식 사이트: <a href="https://codeit.kr/" target="_blank" rel="noopener">https://codeit.kr/</a>)
</div>
