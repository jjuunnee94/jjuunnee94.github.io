---
title: "PyTorch Basic - Autograd"
date: 2026-10-28 17:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, autograd, backward, mse, gradient-descent, sgd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
---

### 1. 예측 반복 과정
```
예측 → 손실 → 기울기 → 파라미터 갱신 → 기울기 초기화 → 새로운 예측
```
### 2. 손실 (loss), 평균 제곱 오차 (MSE)
- 예측 = `가중치(w) x 압력값(x) + 편향(b)`
  
- 오차 = `예측 − 정답(y)`

- 손실 = `(오차)²`

- 평균 제곱 오차(MSE) = `(오차²의 합) ÷ 오차 개수`

### 3. 기울기 (Gradient)

- 손실에서의 기울기
    - **부호** : `−`면 가중치(w)를 올릴수록 손실이 줄고, `+`면 내릴수록 줄어든다.
    - **크기** : 클수록 가중치 변화에 따른 손실(오차)의 변화율이 크다
    
### 4. 경사 하강법 (SGD)

```
새 w = 현재 w − 학습률(lr) × 기울기
```
- 기울기가 `-`일때 (학습률(lr) x 기울기)앞에 `-`를 두어 가중치(w)가 증가하므로 손실이 줄어든다.

- 새 `w`로 기울기를 다시 구하고 또 갱신하는 과정을 **반복**하는 것이 학습이다.

### 5. Autograd (기울기 자동 계산)

- **Autograd(자동 미분)**는 계산 과정을 기록해 두었다가 기울기를 대신 구해 주는 기능이다.

```python
import torch

w = torch.tensor(3.0, requires_grad=True)
x = 2.0
y = 10.0

pred = w * x
loss = (pred - y) ** 2
print(pred)
print(loss)

loss.backward()
print(w.grad)
```

```
tensor(6., grad_fn=<MulBackward0>)
tensor(16., grad_fn=<PowBackward0>)
tensor(-16.)
```

- `requires_grad=True`: "이 값의 기울기가 필요하니 계산 과정을 기록해 줘"라는 표시이다. 손잡이(`w`, `b`)에만 붙인다.
- `grad_fn=...`: 이 값이 어떤 계산(곱셈 `Mul`, 거듭제곱 `Pow`)으로 만들어졌는지의 기록이다.
- `loss.backward()`: 기록을 **거꾸로** 따라가며 기울기를 구해 `.grad`에 넣는다.
- `backward()` 전에는 `.grad`가 `None`이고, 직접 만든 손잡이(leaf)에만 저장된다. `pred`, `loss` 같은 중간 값에는 저장되지 않는다.
- `backward()`는 기울기만 채울 뿐, **`w`의 값은 바꾸지 않는다.** 값을 바꾸는 건 별개의 갱신 단계이다.
- `w.detach()`를 쓰면 계산 기록 없이 값만 볼 수 있다.

### 6. 기울기 공식은 어디서 나올까

- 오차를 `e = w·x + b − y`, 손실을 `L = e²`(데이터 하나)라고 하자.
- `w`를 아주 조금(`δ`) 올리면 오차는 `x·δ`만큼 변한다.

```
e(w + δ) = e + x·δ

L(w + δ) = (e + x·δ)²
         = e² + 2·e·x·δ + x²·δ²
         ≈ e² + 2·e·x·δ          (δ가 작으면 δ² 항은 무시)
```

- 변화량을 `δ`로 나누면 기울기가 나온다. `b`는 `x` 대신 `1`이 붙는다.

```
w의 기울기 = 2 × 오차 × x
b의 기울기 = 2 × 오차
```

- `2`는 제곱을 펼칠 때 생기는 가운데 항 `2·e·(변화량)`에서 나온다.
- 데이터가 `n`개면 손실이 평균이므로 기울기도 평균을 낸다.

```
w의 기울기 = 평균( 2 × 오차 × x )
b의 기울기 = 평균( 2 × 오차 )
```

- 확인: `x = 2`, `w = 3`, `y = 10`이면 오차는 `−4`이고 `2 × (−4) × 2 = −16`이다. `backward()`가 준 값과 같다.
- 이 공식을 외울 필요는 없다. `backward()`가 대신 계산해 주므로, 기울기에는 **틀린 정도**와 **그 손잡이가 오차에 미치는 영향**이 곱해진다는 감각만 있으면 된다.
- `w`와 `b`는 예측에 미치는 영향이 달라서 기울기를 **각각** 구한다. 한쪽은 내리고 다른 쪽은 올려야 하는 상황도 생긴다.

### 7. 데이터가 여러 개일 때와 손실이 값 하나여야 하는 이유

```python
x = torch.tensor([[1.], [2.], [3.]])      # (3, 1)
y = torch.tensor([[5.], [8.], [11.]])     # (3, 1)

w = torch.tensor([[0.]], requires_grad=True)   # (1, 1)
b = torch.zeros(1, requires_grad=True)         # (1,)

pred = x @ w + b                          # (3, 1)
loss = ((pred - y) ** 2).mean()
print(loss.item(), tuple(loss.shape))     # 70.0 ()

loss.backward()
print(w.grad)
print(b.grad)
```

```
70.0 ()
tensor([[-36.]])
tensor([-16.])
```

- `x @ w`는 `(3, 1) @ (1, 1) → (3, 1)`이고, `+ b`는 브로드캐스팅이다.
- `.mean()`이 오차 제곱 3개를 **값 하나**(shape `()`)로 줄인다. `.item()`은 이 값을 파이썬 숫자로 꺼낸다.
- `backward()`는 값 하나로 줄어든 손실에서 출발해야 한다. 값이 여러 개면 어느 것을 줄일지 정해지지 않아서 에러가 난다.

```python
(pred - y).backward()
# RuntimeError: grad can be implicitly created only for scalar outputs
```

- 위 기울기를 공식으로 확인하면 아래와 같다.

```
오차 = −5, −8, −11,  x = 1, 2, 3
w: 평균(−10, −32, −66) = −108 ÷ 3 = −36
b: 평균(−10, −16, −22) = −48 ÷ 3 = −16
```

- 특성이 하나뿐일 때는 `w`를 `(1,)`로 두고 `x * w + b`로 쓰기도 한다. 이 경우 `(3, 1) * (1,)`이 브로드캐스팅으로 같은 결과가 나온다.

### 8. 손잡이 갱신하기: `no_grad`와 기울기 누적

```python
lr = 0.05
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad

print(w, b)
```

```
tensor([[1.8000]], requires_grad=True) tensor([0.8000], requires_grad=True)
```

- `0 − 0.05 × (−36) = 1.8`, `0 − 0.05 × (−16) = 0.8`이다.
- 갱신은 손실을 만드는 식의 일부가 아니라 손잡이를 고치는 동작이다. 이 동작까지 기록하면 기울기가 이전 바퀴까지 거슬러 올라가 값이 섞이고 메모리도 계속 늘어난다.
- 그래서 `with torch.no_grad():` 안에서 "이 계산은 기록하지 마"라고 하고 갱신한다.
- 갱신 후에 예전 `pred`를 출력하면 옛 값 그대로이다. `pred`는 계산해 둔 결과일 뿐 `w`가 바뀐다고 자동으로 다시 계산되지 않는다.

**기울기는 누적된다**

- `backward()`를 부를 때마다 `.grad`에 값이 **더해진다.** (한 번 쓴 기울기를 지우지 않으면 다음 바퀴 기울기와 섞인다.)
- 5절의 단일 데이터 예시로 보자. `w = 3`에서 `.grad`는 `−16`이었다.

```python
lr = 0.05
with torch.no_grad():
    w -= lr * w.grad        # w: 3 → 3.8
pred = w * x                # 새 예측 7.6
loss = (pred - y) ** 2      # 5.76
loss.backward()
print(w.grad)               # tensor(-25.6000)   ← 새 기울기 −9.6이 아니라 −16 + (−9.6)
```

- 그래서 한 바퀴가 끝나면 `.grad`를 **비워야** 한다.

```python
w.grad.zero_()
```

- 비우는 것은 `.grad`뿐이고 `w`의 값은 0이 되지 않는다. 학습이 끝난 뒤 남는 것은 기울기가 아니라 바뀐 `w`, `b`이다.
- 손실 변화를 보고 싶으면 `losses.append(loss.item())`처럼 파이썬 리스트에 직접 담아 둔다.
- 매 바퀴 `pred`와 `loss`를 새로 계산하는 건 정상이다. 같은 `loss`로 `backward()`를 두 번 부르면 `Trying to backward through the graph a second time` 에러가 난다. 누적은 새로 계산한 그래프로 `backward()`를 여러 번 부를 때의 이야기이다.

### 9. SGD 옵티마이저

- 손으로 쓴 `no_grad` 갱신과 `zero_grad`를 대신해 주는 도구이다.
- **SGD(확률적 경사 하강법)**는 4절의 `w − lr × 기울기`를 그대로 적용한다. (데이터를 나눠 쓰는 방식과 이름이 연결되는데, 이 부분은 DataLoader를 다루는 편에서 정리한다.)

```python
w = torch.tensor([[0.]], requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.05)

for epoch in range(1, 5):
    optimizer.zero_grad()                       # ① 기울기 비우기
    pred = x @ w + b                            # ② 예측
    loss = ((pred - y) ** 2).mean()             # ③ 손실
    loss.backward()                             # ④ 기울기 구하기
    optimizer.step()                            # ⑤ 손잡이 돌리기
    print(epoch, f"{loss.item():.4f}")
```

```
1 70.0000
2 13.9200
3 2.7963
4 0.5891
```

| 손으로 쓴 코드 | 옵티마이저 |
| --- | --- |
| `with torch.no_grad(): w -= lr * w.grad` | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

- 출력된 손실은 그 바퀴에서 **갱신하기 전** 값이다.
- `zero_grad()`는 `backward()` **전**에 한다. `backward()`와 `step()` 사이에서 비우면 기울기가 0이 되어 `step()`이 아무것도 바꾸지 못한다.
- `zero_grad(set_to_none=True)`는 기울기를 0이 아니라 `None`으로 비우는 옵션이다. 기본 SGD에서는 결과가 같다.
- 손으로 한 결과와 SGD 결과가 같은지는 `torch.allclose(a, b)`로 확인할 수 있다. 소수 계산은 아주 작은 오차가 생겨서 `==` 대신 쓴다.

| 바퀴 | 갱신 전 손실 | `w` | `b` |
| --- | --- | --- | --- |
| 1 | 70.0 | 0.0 | 0.0 |
| 2 | 13.92 | 1.8 | 0.8 |
| 3 | 2.80 | 2.6 | 1.16 |
| 4 | 0.59 | 2.95 | 1.32 |

- 이 데이터는 `y = 3x + 2`로 만든 것이라, 손실이 줄면서 `w`는 `3`, `b`는 `2`에 가까워진다.

### 10. 정리

| 단계 | 코드 | 하는 일 |
| --- | --- | --- |
| 예측 | `pred = x @ w + b` | 손잡이로 예측을 만든다 |
| 손실 | `((pred - y) ** 2).mean()` | 틀린 정도를 값 하나로 요약한다 |
| 기울기 | `loss.backward()` | `.grad`에 기울기를 채운다 (값은 안 바뀜) |
| 갱신 | `optimizer.step()` | `w − lr × 기울기`로 값을 바꾼다 |
| 비우기 | `optimizer.zero_grad()` | 다음 `backward()` 전에 `.grad`를 비운다 |

- 기울기는 **방향(부호)과 크기**를 알려 주고, 학습률이 한 걸음의 크기를 정한다.
- `backward()`는 기울기만, `step()`은 값만 바꾼다. 둘은 별개의 단계이다.
- 기울기는 누적되므로 매 바퀴 `backward()` 전에 비운다.
