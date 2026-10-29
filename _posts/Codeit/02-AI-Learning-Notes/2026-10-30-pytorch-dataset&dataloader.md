---
title: "PyTorch Basic - Dataset과 DataLoader"
date: 2026-10-29 17:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, dataset, dataloader, batch, epoch, shuffle, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
---

**용어 먼저!**

- **표본(sample)** : 데이터 한 줄
- **batch** : 한 번에 모델에 넣는 표본 묶음
- **batch_size** : batch 하나에 들어가는 표본 수
- **step** : batch 하나를 수행하는 것 ( 예측 → 손실 → 역전파 → 갱신 )
- **epoch** : 전체 표본을 한 바퀴 다 사용한 것
  
### 1. batch와 Dataset

```
표본:  ①  ②  ③  ④  ⑤  ⑥
batch_size = 2

batch 1: ① ②
batch 2: ③ ④
batch 3: ⑤ ⑥
```

- 한 epoch = batch 3개 = **3 step**
- **`Dataset`**: 표본을 보관하고 `i`번째 표본을 꺼내는 규칙을 가진 객체
- **`DataLoader`**: `Dataset`에서 표본을 `batch_size`개씩 묶어 차례로 꺼내는 도구
- `x`, `y`가 이미 Tensor이면 `TensorDataset`으로 짝지어 쓴다.

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor([[1.], [2.], [3.], [4.], [5.], [6.]])           # (6, 1)
y = torch.tensor([[5.], [8.], [11.], [14.], [17.], [20.]])       # (6, 1)

ds = TensorDataset(x, y)
print(len(ds))
print(ds[0])
```

```
6
(tensor([1.]), tensor([5.]))
```

- `len(ds)`: 표본 개수 (`6`)
- `ds[0]`: 0번 표본 하나. `(x 한 줄, y 한 줄)` 짝이다.
- 표본 번호는 0부터 시작하므로 마지막 표본은 `ds[5]`이다.

### 2. DataLoader

```python
loader = DataLoader(ds, batch_size=4, shuffle=False)
for xb, yb in loader:
    print(xb.tolist(), yb.tolist())
```

```
[[1.0], [2.0], [3.0], [4.0]] [[5.0], [8.0], [11.0], [14.0]]
[[5.0], [6.0]] [[17.0], [20.0]]
```
| | batch. 1 | batch. 2 |
| --- | --- | --- |
| xb | [[1.0], [2.0], [3.0], [4.0]]    |  [[5.0], [6.0]] |
| yb | [[5.0], [8.0], [11.0], [14.0]]  | [[17.0], [20.0]] |
- 마지막 batch는 남은 표본
- shape는 `(표본 수, 특성 수)`이다. 위 예시는 `(4, 1)`과 `(2, 1)`이다.

### 3. shuffle과 drop_last

**`shuffle=True`**  : 매 epoch마다 표본 순서를 무작위로 섞어서 batch를 만든다.  
**`shuffle=False`** : 매 epoch마다 표본 순서를 순서대로 batch를 만든다.  

**`drop_last=True`**  : 표본 수가 `batch_size`로 나누어떨어지지 않을 때 마지막 남은 batch를 버린다.  
**`drop_last=False`** : 표본 수가 `batch_size`로 나누어떨어지지 않을 때 마지막 남은 batch를 그대로 사용한다.  

- 미작성시 둘 다 기본값은 `False`
- `shuffle=True`와 `drop_last=True`를 함께 쓰면 버려지는 표본이 epoch마다 달라진다.

### 4. Dataset 만들기

- 데이터가 파일을 꺼낼 때 가공이 필요하면 `Dataset`을 직접 정의한다.

```python
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

ds = MyData(x, y)
print(len(ds))
print(ds[2])
```

```
6
(tensor([3.]), tensor([11.]))
```

| 메서드 | 역할 | 실행되는 때 |
| --- | --- | --- |
| `__init__` | 데이터를 받아 보관 | `MyData(x, y)`로 객체를 만들 때 |
| `__len__` | 표본 개수 반환 | `len(ds)`를 호출할 때 |
| `__getitem__` | `i`번째 표본 반환 | `ds[i]`를 호출할 때 |

- `len(ds)` → `ds.__len__()`
- `ds[2]` → `ds.__getitem__(2)`

### 5. DataLoader로 학습하기

```python
from torch import nn

x = torch.tensor([[1.], [2.], [3.], [4.], [5.], [6.]])
y = torch.tensor([[5.], [8.], [11.], [14.], [17.], [20.]])
loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1, 3):
    for step, (xb, yb) in enumerate(loader, start=1): # enumerate(loader, start=1) : 반복 번호를 1부터 붙인다.
        optimizer.zero_grad()
        pred = model(xb)
        loss = ((pred - yb) ** 2).mean()
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch} step {step} | loss {loss.item():.4f} | "
              f"w {model.weight.item():.4f} b {model.bias.item():.4f}")
```

```
epoch 1 step 1 | loss 44.5000 | w 0.2100 b 0.1300
epoch 1 step 2 | loss 137.3192 | w 1.0384 b 0.3627
epoch 1 step 3 | loss 155.3699 | w 2.4151 b 0.6112
epoch 2 step 1 | loss 5.2210 | w 2.4860 b 0.6565
epoch 2 step 2 | loss 9.9413 | w 2.7085 b 0.7194
epoch 2 step 3 | loss 8.3368 | w 3.0272 b 0.7771
```
- Loss는 갱신 전 값이고 w, b는 step() 이후 값

비교 : **첫 step 직접 계산** (batch: `x = 1, 2` / `y = 5, 8`, `w = b = 0`)

```
예측 = 0, 0 → 오차 = −5, −8 → 손실(평균) = (25 + 64) ÷ 2 = 44.5

w 기울기 = 평균( 2×(−5)×1, 2×(−8)×2 ) = 평균(−10, −32) = −21
b 기울기 = 평균( 2×(−5),   2×(−8)   ) = 평균(−10, −16) = −13

새 w = 0 − (lr)0.01 × (−21) = 0.21
새 b = 0 − (lr)0.01 × (−13) = 0.13
```

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
