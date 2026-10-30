---
title: "PyTorch Basic - 요약정리 01. ~ 05. by AI"
date: 2026-10-30 16::00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, review, autograd, nn-module, nn-linear, dataloader]
---

### 1. 지금까지의 학습 코드 한눈에 보기

- 앞의 글에서 배운 도구를 모두 사용한 학습 코드이다.
- 줄마다 `# 원래:` 뒤에 **도구를 쓰기 전에 직접 작성했던 코드**를 적었다.

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

x = torch.tensor([[1.], [2.], [3.], [4.], [5.], [6.]])
y = torch.tensor([[5.], [8.], [11.], [14.], [17.], [20.]])

loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)
# 원래: x, y 전체를 한 번에 사용

model = nn.Linear(1, 1)
# 원래: w = torch.tensor([[0.]], requires_grad=True)
#       b = torch.zeros(1, requires_grad=True)
with torch.no_grad():
    model.weight.zero_()
    model.bias.zero_()
# 원래: 처음부터 0으로 직접 생성 (nn.Linear는 무작위 초기화라서 0으로 맞춤)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# 원래: SGD([w, b], lr=0.01)  →  그 전에는 optimizer 없이 직접 갱신

for epoch in range(1, 3):
    for step, (xb, yb) in enumerate(loader, start=1):
        # 원래: epoch마다 x, y 전체로 한 번 학습

        optimizer.zero_grad()
        # 원래: w.grad.zero_()
        #       b.grad.zero_()

        pred = model(xb)
        # 원래: pred = x @ w + b

        loss = ((pred - yb) ** 2).mean()
        # 원래: 같음 (손실 식은 도구로 대체하지 않고 직접 작성)

        loss.backward()
        # 원래: 기울기 공식을 직접 계산
        #       w의 기울기 = 평균( 2 × 오차 × x )
        #       b의 기울기 = 평균( 2 × 오차 )

        optimizer.step()
        # 원래: with torch.no_grad():
        #           w -= lr * w.grad
        #           b -= lr * b.grad

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

- `Loss`는 갱신 전 값이고 `w`, `b`는 `step()` 이후 값이다.

### 2. 무엇이 무엇으로 대체됐나

| 현재 코드 | 원래 코드 | 정리한 글 |
| --- | --- | --- |
| `loader`의 `xb`, `yb` | `x`, `y` 전체 | Dataset과 DataLoader |
| `nn.Linear(1, 1)` | `w`, `b`를 직접 생성 | nn.Module과 nn.Linear |
| `model(xb)` | `x @ w + b` | nn.Module과 nn.Linear |
| `model.parameters()` | `[w, b]` 직접 나열 | nn.Module과 nn.Linear |
| `loss.backward()` | 기울기 공식 직접 계산 | Autograd |
| `optimizer.step()` | `no_grad` 안에서 `w -= lr * w.grad` | Autograd |
| `optimizer.zero_grad()` | `w.grad.zero_()` | Autograd |

- 대체되지 않은 것: **손실 식** `((pred - yb) ** 2).mean()`
    - 파이토치의 `nn.MSELoss()`로도 같은 계산을 할 수 있다.

### 3. 한 step의 순서

```
optimizer.zero_grad()   기울기 초기화
pred = model(xb)        예측
loss = ...              손실
loss.backward()         기울기 계산
optimizer.step()        파라미터 갱신
```

- `zero_grad()`는 `backward()` 전에 한다. `backward()`와 `step()` 사이에서 비우면 기울기가 0이 되어 `step()`이 파라미터를 바꾸지 못한다.
- 한 epoch에 `step()`이 일어나는 횟수는 batch 개수와 같다. 위 예시는 6개 ÷ 2 = 3번이다.

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것을 AI로 요약한 내용입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
