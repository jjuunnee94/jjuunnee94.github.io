---
title: "PyTorch Basic - 07. device와 CrossEntropyLoss"
date: 2026-11-02 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, device, cuda, crossentropyloss, softmax, logits, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
---

### 1. device : 계산할 장치 정하기

- **CPU**: 모든 컴퓨터에 있는 기본 계산 장치
- **GPU**: 같은 계산을 동시에 많이 처리하는 장치
- PyTorch에서는 장치를 `device`로 표현한다.

| 이름 | 뜻 |
| --- | --- |
| `"cpu"` | CPU |
| `"cuda"` | NVIDIA GPU |

GPU가 없는 컴퓨터도 많으므로, "GPU가 있으면 GPU, 없으면 CPU"로 정하는 코드를 기본으로 쓴다.

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
```

```
cpu
```

- `torch.cuda.is_available()`: GPU(CUDA)를 쓸 수 있으면 `True`, 아니면 `False`
- `torch.device("...")`: 장치 이름을 `device` 객체로 만든다.
- 위 출력은 GPU가 없는 환경의 결과이며, GPU가 있으면 `cuda`가 출력된다.

### 2. .to(device) : Tensor와 모델의 차이

- 계산에 쓰는 Tensor와 모델은 **같은 장치**에 있어야 한다. 장치가 다르면 오류가 난다.
- 장치를 옮기는 함수는 `.to(device)`이지만, Tensor와 Module에서 동작이 다르다.

| 대상 | `.to(device)`의 동작 | 사용법 |
| --- | --- | --- |
| Tensor | 옮긴 **새 Tensor를 반환**한다. 원래 Tensor는 그대로이다. | `x = x.to(device)` |
| Module(모델) | 파라미터를 **제자리에서 옮기고** 자기 자신을 반환한다. | `model.to(device)` |

```python
from torch import nn

x = torch.tensor([[1.], [2.]])
x.to("cuda")                 # 반환값을 담지 않았으므로 x는 그대로 CPU
x = x.to("cuda")             # 반환값을 담아야 x가 GPU로 옮겨진다

model = nn.Linear(1, 1)
model.to("cuda")             # 이것만으로 model.weight가 GPU로 옮겨진다
```

- 위 코드는 GPU가 있는 환경을 가정한 것이다.
- Tensor는 `x = x.to(device)`처럼 **반환값을 받아야** 하고, 모델은 `model.to(device)`만 호출해도 된다.

**순서 규칙**

- `optimizer`는 만들 때 받은 파라미터를 기억하므로, 모델을 장치로 옮긴 **뒤에** 만든다.

```python
model = nn.Linear(1, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

**DataLoader와 함께 쓸 때**

- DataLoader는 반복할 때마다 CPU의 배치를 새로 꺼내 준다.
- 그러므로 반복문 안의 맨 윗줄에서 `x`와 `y`를 **둘 다** 옮긴다.
  - `x`는 `model(x)`에서 모델과 만난다.
  - `y`는 `loss_fn(pred, y)`에서 `pred`와 만난다.

```python
for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)

    pred = model(xb)
    loss = loss_fn(pred, yb)
```

### 3. 회귀와 분류

- **회귀**: 연속된 **숫자**를 예측한다. 손실은 MSE를 쓴다.
- **분류**: 정해진 **범주**를 맞힌다. (예: 고양이 / 개 / 새)

| 용어 | 뜻 | 예 |
| --- | --- | --- |
| class | 구별할 범주 | 고양이, 개, 새 |
| label | 표본의 정답 범주 | 이 사진은 "개" |
| class index | 범주를 가리키는 0부터의 번호 | 고양이 0, 개 1, 새 2 |

- 컴퓨터는 이름을 다루지 못하므로, 범주를 번호(class index)로 바꾸고 정답 `label`도 이 번호로 쓴다.

### 4. logits : class마다 하나씩 나오는 점수

- 분류 모델의 마지막 `Linear`는 **class 개수만큼** 점수를 출력한다. 이 점수를 **logits**라고 한다.
  - class가 3개이면 `nn.Linear(?, 3)`이고, 회귀처럼 출력이 1개가 아니다.
- 점수가 클수록 그 class일 가능성이 높다고 본다.
- 가장 큰 칸의 번호를 찾는 함수가 `argmax`이다.

```python
logits = torch.tensor([[0.2, 1.5, -0.3],
                       [3.0, 0.1,  2.9]])

print(logits.shape)
print(logits.argmax(dim=1))
```

```
torch.Size([2, 3])
tensor([1, 0])
```

- `logits`의 shape는 `(N, C)`이다. 행은 표본(N개), 열은 class(C개)이다.
- `dim=1`은 각 행에서 열 방향으로 가장 큰 칸을 찾으라는 뜻이다.
  - 첫 행은 `1.5`가 1번 칸, 둘째 행은 `3.0`이 0번 칸이므로 예측은 `[1, 0]`이다.

### 5. softmax : 점수를 확률로 바꾸기

- 모든 값을 양수이면서 합이 1인 확률로 만든다.

```
class i의 확률 = exp(점수 i) ÷ (모든 class의 exp(점수)의 합)
```

- `exp(a)`는 `e^a`(e ≈ 2.718)이며, 어떤 수를 넣어도 결과가 양수이다.
- 모든 값을 `exp`한 합으로 나누므로 확률의 합이 1이 된다.
- 점수가 클수록 `exp`가 더 크게 늘어나므로 확률도 높아진다.

**예시** : `logits = [1.0, 2.0, 0.0]`

| class | 점수 | `exp(점수)` | 확률 (÷ 11.107) |
| --- | --- | --- | --- |
| 0 | 1.0 | 2.718 | 0.245 |
| 1 | 2.0 | 7.389 | 0.665 |
| 2 | 0.0 | 1.000 | 0.090 |
| 합 | | 11.107 | 1.000 |

```python
logits = torch.tensor([1.0, 2.0, 0.0])
print(torch.softmax(logits, dim=0))
```

```
tensor([0.2447, 0.6652, 0.0900])
```

### 6. CrossEntropyLoss

- 정답 확률에 따라 벌점을 매김.

```
손실 = −ln(정답 class의 확률)
```

- `ln`은 자연로그이다.

| 정답(label) | 정답 확률 | 손실 `−ln(확률)` |
| --- | --- | --- |
| 1 | 0.665 | 0.408 |
| 0 | 0.245 | 1.408 |
| 2 | 0.090 | 2.408 |

- 정답에 높은 확률을 줄수록 손실이 작아지고, 낮은 확률을 줄수록 손실이 커진다.
- 정답 확률이 1이면 `−ln(1) = 0`이다.
- softmax → 정답 확률 → −ln 전체를 `nn.CrossEntropyLoss` 하나가 처리한다.
- softmax는 함수 안에서 처리하므로, `logits`를 **그대로** 넣는다. 확률로 바꾼 값을 넣으면 softmax가 두 번 적용된다.

```python
from torch import nn

logits = torch.tensor([[1.0, 2.0, 0.0],
                       [3.0, 0.1, 2.9]])      # (2, 3)
labels = torch.tensor([1, 0])                 # (2,)

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, labels)

print(loss.item(), loss.shape)
```

```
0.5402 torch.Size([])
```

| 표본 | 정답 | 정답 확률 | 손실 |
| --- | --- | --- | --- |
| 첫 행 | 1 | 0.665 | 0.4076 |
| 둘째 행 | 0 | 0.510 | 0.6729 |

- 표본별 손실의 **평균** `(0.4076 + 0.6729) ÷ 2 ≈ 0.5402`가 결과이다.
- MSE처럼 평균을 낸 스칼라 하나이므로 `backward()`를 바로 호출할 수 있다.

### 7. 회귀(MSE)와 분류(CrossEntropyLoss)의 차이

| | 회귀 (MSE) | 분류 (CrossEntropyLoss) |
| --- | --- | --- |
| 모델 출력 | `(N, 1)` 숫자 | `(N, C)` class별 점수(logits) |
| 정답 | `(N, 1)`, `float` | `(N,)`, `long` (class 번호) |
| 손실 | 오차 제곱 평균 | `−ln(정답 확률)` 평균 |

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
