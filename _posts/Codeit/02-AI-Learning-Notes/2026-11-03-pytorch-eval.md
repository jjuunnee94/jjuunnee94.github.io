---
title: "PyTorch Basic - 평가와 모델 저장"
date: 2026-11-03 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, eval, no-grad, state-dict, torch-save, allclose, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
---
 
### 1. 학습과 평가

- **학습**: `forward → loss → backward → step`으로 파라미터를 고친다.
- **평가**: 완성된 모델을 새 데이터에 시험해 보는 일이며, 파라미터를 **고치지 않고** 예측과 손실만 확인한다.
- 평가에는 학습에 쓰지 않은 **검증(validation) 데이터**를 쓴다. 코드의 `x_val`, `y_val`이 이 데이터이다.

### 2. no_grad : 계산 기록 끄기

- `requires_grad=True`인 Tensor로 계산하면 PyTorch는 `backward()`를 위해 계산 기록(`grad_fn`)을 남긴다.
- 평가에서는 `backward()`를 부르지 않으므로 기록이 메모리와 시간만 쓴다. 그래서 `torch.no_grad()`로 기록을 끈다.

```python
import torch
from torch import nn

model = nn.Linear(1, 2)
x = torch.tensor([[1.5]])

pred = model(x)
print(pred.requires_grad)

with torch.no_grad():
    pred2 = model(x)
print(pred2.requires_grad)
```

```
True
False
```

- `with torch.no_grad():` 블록 안의 계산은 기록이 남지 않는다. 블록을 벗어나면 다시 기록이 켜진다.
- 기록이 남든 안 남든 계산 값 자체는 같다.

### 3. eval() 과 no_grad 의 차이

- `model.eval()` / `model.train()`: 모델의 **모드**를 바꾼다. Dropout, BatchNorm 같은 층은 모드에 따라 동작이 달라지며, 지금 쓰는 `Linear`와 `ReLU`는 같다.

| 함수 | 하는 일 | 하지 않는 일 |
| --- | --- | --- |
| `model.eval()` | 모델을 평가 모드로 바꾼다. | gradient 기록을 끄지 않는다. |
| `torch.no_grad()` | gradient 기록을 끈다. | 모델의 모드를 바꾸지 않는다. |

- 둘은 별개이므로 평가할 때는 보통 같이 쓴다.

```python
model.eval()
with torch.no_grad():
    val_pred = model(x_val)
```

- `zero_grad()`와도 다르다. `zero_grad()`는 쌓인 `.grad` 값을 비우고, `no_grad`는 새 기록을 남기지 않는다.

### 4. detach().clone() : 평가 전 값 보관하기

- 평가 전후에 파라미터가 변하지 않았는지 확인하려면, 평가 전 값을 복사해 두어야 한다.
- `detach()`: 계산 기록에서 분리하지만 원래 Tensor와 **저장 공간을 공유**한다.
- `clone()`: 값을 **복사**해 새 저장 공간에 둔다.

```python
w = nn.Parameter(torch.tensor([2.0]))

a = w.detach()            # 저장 공간 공유
b = w.detach().clone()    # 값 복사

with torch.no_grad():
    w.add_(1.0)           # w를 3.0으로 변경

print(a)
print(b)
```

```
tensor([3.])
tensor([2.])
```

- `a`는 `w`와 저장 공간을 공유하므로 `w`의 변화가 그대로 보이고, `b`는 복사본이라 변하지 않는다.
- 그러므로 비교용 기록은 `detach().clone()`으로 남긴다.

### 5. 평가 전체 흐름

- 아래 모델은 학습하지 않고 **직접 정한 값**을 넣은 것이며, 평가 경로를 보기 위한 예시이다.

```python
model = nn.Linear(1, 2)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[-1.], [1.]]))
    model.bias.zero_()

x_val = torch.tensor([[-1.5], [1.5]])      # 검증 표본 2개
y_val = torch.tensor([0, 1])               # 정답 class 번호

before = {k: v.detach().clone() for k, v in model.named_parameters()}

model.eval()
with torch.no_grad():
    val_logits = model(x_val)
    val_loss = nn.CrossEntropyLoss()(val_logits, y_val)

unchanged = all(torch.equal(before[k], v) for k, v in model.named_parameters())
print(val_logits)
print(val_loss.item(), unchanged)
```

```
tensor([[ 1.5000, -1.5000],
        [-1.5000,  1.5000]])
0.0486 True
```

- `before`: 평가 전 파라미터를 복사해 둔 사전이다.
- `torch.equal(a, b)`: 두 Tensor의 모양과 값이 완전히 같으면 `True`이다.
- `unchanged`가 `True`이면 평가가 파라미터를 바꾸지 않았다는 확인이다. 파라미터를 바꾸는 것은 `optimizer.step()`이다.
- 손실이 작은 이유: 두 표본 모두 정답 class의 점수가 `1.5`, 다른 쪽이 `−1.5`이므로 정답 확률이 약 95%이다. 손실은 `−ln(e^1.5 ÷ (e^1.5 + e^−1.5)) ≈ 0.0486`이다.

### 6. state_dict : 모델의 상태 저장하기

- 학습한 모델을 저장하지 않으면 프로그램이 끝나는 순간 파라미터가 사라진다.
- 저장하는 대상은 모델 객체 전체가 아니라 **`state_dict`**이다.
  - 모델의 등록된 이름과 Tensor를 담은 사전(dict)이다.
  - **`forward` 코드는 저장하지 않고** 값만 저장한다.

```python
model = nn.Linear(1, 2)
sd = model.state_dict()

print(type(sd))
print({k: tuple(v.shape) for k, v in sd.items()})
```

```
<class 'collections.OrderedDict'>
{'weight': (2, 1), 'bias': (2,)}
```

- 키(`weight`, `bias`)는 `named_parameters()`의 이름과 같다.

### 7. 저장과 복원

- 파일 **읽기**는 파일만 있으면 되지만, **예측에 쓰려면** 같은 구조의 모델 객체에 값을 채워야 한다. 파일에는 `forward`가 없기 때문이다.
- 새로 만든 모델의 파라미터는 **랜덤 초기값**이며, `load_state_dict`를 호출해야 저장한 값으로 채워진다.
  - 중간층의 칸들이 같은 값으로 시작하면 서로 똑같이 학습되므로, 모델은 작은 랜덤 값으로 시작한다.

```python
model = nn.Linear(1, 2)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[-1.], [1.]]))
    model.bias.zero_()

torch.save(model.state_dict(), "linear_state.pt")       # 1. 저장

restored = nn.Linear(1, 2)                               # 2. 같은 구조의 새 모델
state = torch.load("linear_state.pt", map_location="cpu", weights_only=True)
restored.load_state_dict(state)                          # 3. 값 채우기
restored.eval()
```

| 코드 | 역할 |
| --- | --- |
| `torch.save(obj, 파일)` | 객체(여기서는 `state_dict`)를 파일에 저장한다. |
| `torch.load(파일)` | 파일에서 읽어 온다. |
| `map_location="cpu"` | 읽은 Tensor를 CPU에 올린다. GPU에서 저장한 파일을 GPU 없는 컴퓨터에서 열 때 필요하다. |
| `weights_only=True` | 가중치 같은 제한된 종류의 객체만 읽는다. (안전을 위한 설정) |
| `load_state_dict(state)` | 새 모델에 값을 채운다. |

- `load_state_dict`는 모델 객체를 반환하지 않으므로 `restored = restored.load_state_dict(state)`처럼 덮어쓰지 않는다.

### 8. allclose : 복원이 잘 되었는지 확인하기

- 복원한 모델과 원래 모델에 **같은 입력**을 넣어 예측이 같으면 복원이 잘 된 것이다.
- 소수 계산에는 아주 작은 오차가 생길 수 있어서 `==` 대신 오차 범위 안에서 가까운지 보는 `torch.allclose`를 쓴다.

```python
model.eval()
with torch.no_grad():
    pred_before = model(x_val)
    pred_after = restored(x_val)

print(torch.allclose(pred_before, pred_after))
```

```
True
```

- 비교는 **같은 입력, 같은 구조, 같은 모드(`eval`), 같은 장치**에서 한다.
- 예측이 같다는 것은 복원이 되었다는 뜻이지, 모델의 성능이 높다는 뜻은 아니다.

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
