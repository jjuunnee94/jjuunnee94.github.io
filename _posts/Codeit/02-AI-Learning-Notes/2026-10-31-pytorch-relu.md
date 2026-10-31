---
title: "PyTorch Basic - ReLU와 MLP"
date: 2026-10-31 17:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, relu, mlp, nn-linear, forward, 비선형, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
---

### 1. 직선 모델의 한계

![직선 모델의 한계 그래프](/assets/weekly-paper/10-31-relu.1.png)

**[ y = x² ]**

| x | -2 | -1 | 0 | 1 | 2 |
|---|---|---|---|---|---|
| y | 4 | 1 | 0 | 1 | 4 |

- x가 음수일때는 기울기가 음수이고 x가 양수일때는 기울기가 양수이므로 기울기가 상쇄되어 기울기 w는 0이 된다.
- w가 0일때 y =0 * x + b 가 되어 b는 y값의 평균인 2가 되어 y = 0 * x + 2 가 된다.

```
예측 = 2, 2, 2, 2, 2
오차(예측 − 정답) = −2, 1, 2, 1, −2
손실(MSE) = (4 + 1 + 4 + 1 + 4) ÷ 5 = 2.8
```
- **1차함수(직선)의 한계**: 직선은 모든 x에서 기울기가 `w` 하나로 고정되어 있어서 곡선을 그릴 수 없다.

### 2. ReLU : 직선을 꺾을 수 있는 도구

- **`ReLU`** :  
  `값` > 0 이면 `값`  
  `값` <= 0 이면 0  

```
ReLU(z) = max(0, z) # z > 0 이면 z / z <= 0 = 0
```

```python
import torch

z = torch.tensor([-3., -0.5, 0., 2., 4.])
print(torch.relu(z))
```

```
tensor([0., 0., 0., 2., 4.])
```

- 그래프로 그리면 왼쪽은 평평하고 0에서 꺾여 오른쪽은 올라가는 선이다. 기울기가 `0`에서 `1`로 바뀌는 **꺾임**이 핵심이다.
- `ReLU`를 끼우면 직선이 꺾인 선이 된다.

| x | -2 | -1 | 0 | 1 | 2 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **`z = x − 1`** | -3 | -2 | -1 | 0 | 1 |
| **`h = ReLU(z)`** | 0 | 0 | 0 | 0 | 1 |
| **`out = 2 × h + 1`** | 1 | 1 | 1 | 1 | 3 |

![변형된 ReLU 그래프와 꺾이는 지점](/assets/weekly-paper/10-31-relu.2.png)

- `x`가 `1`까지는 `out`이 `1`로 평평하다가 `1`을 넘으면 올라간다. `x − 1`이 `0`이 되는 지점에서 꺾인다.
- 꺾이는 지점을 여러 개 만들면 U자 같은 모양도 만들 수 있다.

### 3. MLP: Linear → ReLU → Linear

- 층을 쌓은 모델을 **MLP**(다층 퍼셉트론)라고 한다.

```
입력 x → Linear → ReLU → Linear → 예측
```

- 첫 번째 `Linear`: 입력에서 여러 개의 `z`를 만든다.
- `ReLU`: 각 `z`를 꺾는다. (음수는 0)
- 두 번째 `Linear`: 꺾인 값들을 합쳐 최종 예측을 만든다.

**직접 정한 값으로 U자 만들기**

- 첫 번째 `Linear(1, 2)`: `z₁ = x`, `z₂ = −x`
- `ReLU`: `h₁ = ReLU(x)`, `h₂ = ReLU(−x)`
- 두 번째 `Linear(2, 1)`: `예측 = 1.5 × h₁ + 1.5 × h₂`

```
x       :  −2    −1     0     1     2
z₁ = x  :  −2    −1     0     1     2
z₂ = −x :   2     1     0    −1    −2
h₁      :   0     0     0     1     2
h₂      :   2     1     0     0     0
예측     :   3   1.5     0   1.5     3
정답 y   :   4     1     0     1     4
```

- `h₁`은 오른쪽만, `h₂`는 왼쪽만 올라가서 합치면 양쪽 끝이 높은 U자가 된다.
- 오차는 `−1, 0.5, 0, 0.5, −1`이고 손실은 `(1 + 0.25 + 0 + 0.25 + 1) ÷ 5 = 0.5`이다. 직선의 최선(`2.8`)보다 훨씬 작다.

```python
from torch import nn

class UShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 2)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        z = self.layer1(x)
        h = self.act(z)
        return self.layer2(h)

model = UShape()
with torch.no_grad():
    model.layer1.weight.copy_(torch.tensor([[1.], [-1.]]))
    model.layer1.bias.zero_()
    model.layer2.weight.copy_(torch.tensor([[1.5, 1.5]]))
    model.layer2.bias.zero_()

x = torch.tensor([[-2.], [-1.], [0.], [1.], [2.]])
y = x ** 2
pred = model(x)
loss = ((pred - y) ** 2).mean()
print(pred.shape, f"{loss.item():.4f}")
```

```
torch.Size([5, 1]) 0.5000
```

- `nn.ReLU()`: `ReLU`를 층처럼 쓰는 도구이며 파라미터가 없다.

### 4. 층을 지나는 shape

- 입력 `x`가 `(4, 1)`이고 `Linear(1, 3)` → `ReLU` → `Linear(3, 1)`인 MLP를 보자.

| 단계 | shape | 이유 |
| --- | --- | --- |
| 입력 `x` | (4, 1) | 표본 4개, 특성 1개 |
| `Linear(1, 3)` → `z` | (4, 3) | 표본마다 값이 3개 |
| `ReLU` → `h` | (4, 3) | 값만 바뀌고 모양은 그대로 |
| `Linear(3, 1)` → `pred` | (4, 1) | 표본마다 값 1개로 합침 |

```
x=1 → z=[1, −1, 1]  → h=[1, 0, 1]  → pred = 1×1 + 1×0 + 1×1 = 2
```

- 줄(표본 수)은 층을 지나도 변하지 않는다. 층은 표본 하나하나에 똑같이 적용된다.
- 칸은 `Linear(입력, 출력)`에서 입력 칸 수가 출력 칸 수로 바뀐다.
- `ReLU`는 모양을 바꾸지 않는다.
- 앞 층의 출력 칸 수와 다음 층의 입력 칸 수가 같아야 한다. 행렬곱으로는 `(4, 1) @ (1, 3) → (4, 3)`, `(4, 3) @ (3, 1) → (4, 1)`이다.

### 5. ReLU가 없으면 직선이 된다

- `Linear`를 몇 개 쌓아도 `ReLU` 같은 꺾는 함수가 없으면 결국 직선이다.

```
첫 번째 층:  z   = 2x + 1
두 번째 층:  out = 3z − 4

out = 3 × (2x + 1) − 4 = 6x − 1
```

```
x   :  0    1    2
out : −1    5   11        (1 늘 때마다 6씩 증가)
```

- 층이 두 개여도 합치면 `w × x + b` 꼴의 직선이다.

### 6. 층 선언과 forward 연결

- `__init__`에서 `self.이름 = nn.Linear(...)`로 선언하면 **등록**은 자동으로 된다. 하지만 실제로 계산에 쓰이려면 `forward`에서 호출해야 한다.

```python
class Unused(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(1, 1)
        self.b = nn.Linear(1, 1)    # 선언만 하고 forward에서 사용하지 않음

    def forward(self, x):
        return self.a(x)

m = Unused()
loss = m(torch.tensor([[1.]])).sum()
loss.backward()

for name, p in m.named_parameters():
    print(name, p.grad is None)
```

```
a.weight False
a.bias False
b.weight True
b.bias True
```

- `b`는 등록되어 `named_parameters()`에 나오지만 계산에 쓰이지 않아 기울기가 `None`이다.
- 기울기가 없으면 `optimizer.step()`에서 값이 바뀌지 않는다.
- `self.act = nn.ReLU()`를 선언하고 `forward`에서 `h = self.act(z)`를 빠뜨리면 `Linear` 두 개만 쌓은 직선 모델이 된다. 에러가 나지 않아서 알아채기 어렵다.
- **선언 = 등록**, **forward에서 호출 = 실제 계산**이다.

### 7. MLP 학습하기

- 학습 한 바퀴를 함수로 묶는다.

```python
def train(model, x, y, optimizer, steps):
    history = []
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history
```

```python
model = UShape()
with torch.no_grad():
    model.layer1.weight.copy_(torch.tensor([[0.5], [-0.5]]))
    model.layer1.bias.copy_(torch.tensor([0.1, 0.1]))
    model.layer2.weight.copy_(torch.tensor([[0.5, 0.5]]))
    model.layer2.bias.zero_()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
history = train(model, x, y, optimizer, steps=300)
print([f"{history[i]:.4f}" for i in (0, 1, 2, 9)])
with torch.no_grad():
    print([round(v, 3) for v in model(x).flatten().tolist()])
```

```
['4.9590', '2.9426', '1.7556', '0.5806']
[3.999, 1.002, -0.004, 1.002, 3.999]
```

- 기록된 손실은 그 step에서 갱신 전 값이다.
- 처음 손실 `4.9590`에서 10번째 step에는 `0.58`까지 줄고, 300번 학습한 뒤 예측은 정답(`4, 1, 0, 1, 4`)에 거의 가까워진다.
- 직선 모델의 최선이 `2.8`이었는데 MLP는 거의 `0`까지 내려간다.
- 숫자는 환경에 따라 마지막 자리가 조금 다를 수 있다.

**처음 손실 직접 계산** (`x = −2`)

```
z₁ = 0.5 × (−2) + 0.1 = −0.9
z₂ = −0.5 × (−2) + 0.1 = 1.1
h₁ = 0,  h₂ = 1.1
pred = 0.5 × 0 + 0.5 × 1.1 = 0.55    (정답 4, 오차 −3.45)
```

- 같은 방식으로 `x = 0`에서는 `z₁ = z₂ = 0.1`, `pred = 0.1`, 오차 `0.1`이다. 다섯 점의 오차 제곱 평균이 `4.959`이다.

**초기값과 시드**

- `nn.Linear`는 가중치를 무작위로 초기화하므로 실행마다 결과가 달라진다. 같은 결과를 원하면 모델을 만들기 전에 `torch.manual_seed(0)`으로 시드를 고정한다.
- 시드를 고정해도 파이토치 버전이나 장치가 다르면 같은 숫자를 보장하지는 않는다.

**shape 확인**

- 학습 함수 안에서 예측과 정답의 shape가 같은지 확인하면 안전하다.

```python
if pred.shape != y.shape:
    raise ValueError(f"prediction {pred.shape} and target {y.shape} must match")
```

- `(N, 1)`과 `(N,)`을 섞으면 브로드캐스팅으로 `(N, N)`이 되어 에러 없이 손실이 틀어진다.

### 8. 정리

| 개념 | 설명 |
| --- | --- |
| 직선 모델의 한계 | 기울기가 하나라서 U자 같은 모양을 만들 수 없다 |
| `ReLU` | 음수는 0, 양수는 그대로. 직선을 꺾는다 |
| MLP | `Linear → ReLU → Linear`로 층을 쌓은 모델 |
| 층 shape | 줄(표본 수)은 유지, 칸은 `Linear(입력, 출력)`에서 바뀐다 |
| `ReLU` 없는 층 쌓기 | 합치면 결국 직선 |
| 선언과 forward | 선언은 등록, `forward`에서 호출해야 실제로 계산에 쓰인다 |
| 학습 | 기존 학습 루프(`zero_grad` → 예측 → 손실 → `backward` → `step`) 그대로 |

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
