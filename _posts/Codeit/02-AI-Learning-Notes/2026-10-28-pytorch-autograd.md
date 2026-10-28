---
title: "PyTorch Basic - Autograd"
date: 2026-10-28 09:00:00 +0900
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
    - **부호**: `−`면 가중치(w)를 올릴수록 손실이 줄고, `+`면 내릴수록 줄어든다.
    - **크기**: 클수록 가중치 변화에 따른 손실(오차)의 변화율이 크다

```
w의 기울기 = 평균( 2 × 오차 × x )
b의 기울기 = 평균( 2 × 오차 )
```
    
### 4. 경사 하강법 (SGD)

```
새 w = 현재 w − 학습률(lr) × 기울기
```
- 기울기가 `-`일때 (학습률(lr) x 기울기)앞에 `-`를 두어 가중치(w)가 증가하므로 손실이 줄어든다.

- 새 `w`로 기울기를 다시 구하고 또 갱신하는 과정을 **반복**하는 것이 학습이다.

### 5. 기울기 자동계산 (Autograd)

- **Autograd(자동 미분)**: 계산 과정을 기록해 두었다가 기울기를 대신 구해 주는 기능

```python
import torch

w = torch.tensor(3.0, requires_grad=True) #requires_grad=True: 기울기 계산 과정을 기록, (w, b)에만 쓰임
x = 2.0
y = 10.0

# 순전파
pred = w * x #예측값
loss = (pred - y) ** 2 #손실값
print(pred)
print(loss)

#역전파
loss.backward() #계산된 순서를 거꾸로 따라가며 기울기 계산
print(w.grad)
```

```
tensor(6., grad_fn=<MulBackward0>) # Mul은 곱셈을 통해 계산된 값이므로 역전파 시 곱셈으로 기울기를 구하겠다는 기록
tensor(16., grad_fn=<PowBackward0>) # Pow는 제곱을 통해 계산된 값이므로 역전파 시 제곱으로 기울기를 구하겠다는 기록
tensor(-16.)
```

- `backward()`는 **단 하나의 숫자**여야만 가중치로 전달할 기울기를 계산해냅니다.
- 데이터가 여러 개일 때 `.mean()`이나 `.sum()`을 써서 값 1개로 줄여주지 않으면, 파이토치가 어느 숫자를 기준으로 미분해야 할지 몰라 `RuntimeError` 에러가 발생합니다.

### 6. 기울기 누적

```python
w = 0
b = 0
w.grad = -36
b.grad = -16
lr = 0.05
with torch.no_grad(): # 계산 기록 X
    w -= lr * w.grad
    b -= lr * b.grad

print(w, b)
```

```
tensor([[1.8000]], requires_grad=True) tensor([0.8000], requires_grad=True)
```

- 앞선 공식에 숫자를 대입해 보면 `0 − 0.05 × (−36) = 1.8`, `0 − 0.05 × (−16) = 0.8`이다.
- `with torch.no_grad():` 안에서 갱신된 `w`, `b`의 연산 과정은 그래프에 기록되지 않고 값만 갱신된다.
- 갱신 후에 출력값(`pred`)은 그대로 유지된다. `pred`는 이미 계산해 둔 결과일 뿐, `w`가 바뀐다고 해서 자동으로 다시 계산되지 않는다.

**기울기는 누적된다**

- `backward()`를 부를 때마다 `.grad`에 값이 **더해진다.** (한 번 쓴 기울기를 지우지 않으면 다음 기울기와 합해진다.)

```python
x = 2
y = 10
w = 3
w.grad = -16
lr = 0.05

with torch.no_grad():
    w -= lr * w.grad        # w: 3 → 3.8
pred = w * x                # 새 예측 7.6
loss = (pred - y) ** 2      # 5.76
loss.backward
print(w.grad)               # w.grad = 2 × 오차 × x
```
```
tensor(-25.6000) # 기존 기울기 + 새로운 기울기  → −16 + (−9.6)  = -25.6
```
- 기존 기울기 값이 남아있어 w.grad의 값은 -25.6이 되었다.
- 그래서 한 바퀴가 끝나면 `.grad`를 **초기회** 해야한다.

```python
w.grad.zero_() # w.grad 초기화
```

- 초기화하는 것은 `.grad`뿐이고 `w`의 값은 0이 되지 않는다. 남는 것은 기울기가 아니라 바뀐 `w`, `b`이다.

### 7. SGD 옵티마이저

- 직접 쓴 `no_grad` 갱신과 `zero_grad`를 대신해 주는 기능

```python
x = torch.tensor([[1.], [2.], [3.]])  
y = torch.tensor([[5.], [8.], [11.]])

w = torch.tensor([[0.]], requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.05)

for epoch in range(1, 5):
    optimizer.zero_grad()                       # ① 기울기 비우기
    pred = x @ w + b                            # ② 예측
    loss = ((pred - y) ** 2).mean()             # ③ 손실(평균값)
    loss.backward()                             # ④ 기울기 계산
    optimizer.step()                            # ⑤ 가중치 갱신
    print(epoch, f"{loss.item():.4f}")
```
회차당 손실값
```
회차당 손실값
1 70.0000
2 13.9200
3 2.7963
4 0.5891
```

| 직접 쓴 코드 | 옵티마이저 |
| --- | --- |
| `with torch.no_grad(): w -= lr * w.grad` | `optimizer.step()` |
| `w.grad.zero_()` | `optimizer.zero_grad()` |

- 출력된 손실은 **갱신하기 전** 값이다.
- `zero_grad()`는 `backward()` **전**에 한다.

| 바퀴 | 갱신 전 손실 | `w` | `b` |
| --- | --- | --- | --- |
| 1 | 70.0 | 0.0 | 0.0 |
| 2 | 13.92 | 1.8 | 0.8 |
| 3 | 2.80 | 2.6 | 1.16 |
| 4 | 0.59 | 2.95 | 1.32 |

- 이 데이터는 `y = 3x + 2`로 만든 것이라, 손실이 줄면서 `w`는 `3`, `b`는 `2`에 가까워진다.

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
