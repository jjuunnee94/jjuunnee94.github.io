---
title: "PyTorch Basic - AI와 함께 PyTorch전체 코드 정리하기"
date: 2026-11-03 17:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, pytorch, mlp, dataloader, crossentropyloss, state-dict, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
---

### 1. 전체 코드

- 지금까지 배운 것을 한 번에 연결한 코드이다.
- 문제는 `x`의 절댓값이 1보다 크면 `1`, 아니면 `0`으로 분류하는 것이다. 직선 하나로는 나눌 수 없으므로 `ReLU`가 있는 MLP를 쓴다.
- 주석의 "원래"는 앞 포스팅에서 직접 쓰던 방식이고, 괄호 안 ③~⑧은 해당 포스팅 번호이다.

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# ---------- 준비 ----------
torch.manual_seed(0)                                   # 랜덤 초기값을 고정해서 같은 결과를 재현한다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU가 있으면 GPU, 없으면 CPU를 계산 장치로 정한다.

# ---------- 데이터 ----------
x_train = torch.tensor([[-2.0], [-1.5], [-1.2], [-0.8], [-0.5], [-0.2],
                        [0.2], [0.5], [0.8], [1.2], [1.5], [2.0]])   # 학습용 입력, shape (12, 1)
y_train = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])        # 학습용 정답 class 번호, shape (12,) / |x| > 1이면 1
x_val = torch.tensor([[-1.8], [-0.6], [0.4], [1.7]])                 # 검증용 입력, shape (4, 1)
y_val = torch.tensor([1, 0, 0, 1])                                   # 검증용 정답, shape (4,)

train_loader = DataLoader(TensorDataset(x_train, y_train),           # 같은 번호의 x와 y를 한 쌍으로 묶는다.
                          batch_size=4, shuffle=True)                # 4개씩 묶어서 섞어 가며 꺼내 준다.

# ---------- 모델 ----------
class MLP(nn.Module):                                  # 계산 구조와 파라미터를 하나로 묶는 모델 클래스
    def __init__(self):
        super().__init__()                             # nn.Module의 기능(파라미터 등록 등)을 사용하기 위한 준비
        self.layer1 = nn.Linear(1, 8)                  # 입력 1개 → 임의의 값 8개를 만드는 첫 번째 Linear (w, b가 자동 등록된다.)
        self.act = nn.ReLU()                           # 음수는 0으로 만들어 직선을 꺾는다.
        self.layer2 = nn.Linear(8, 2)                  # 값 8개 → class 2개(y값 0또는1)의 점수(logits)를 만드는 두 번째 Linear

    def forward(self, x):                              # model(x)를 호출하면 실행되는 계산 순서
        z = self.layer1(x)                             # (N, 1) → (N, 8)
        h = self.act(z)                                # 모양은 그대로, 음수만 0이 된다. (N, 8)
        return self.layer2(h)                          # (N, 8) → (N, 2), softmax를 붙이지 않은 점수 그대로 반환한다.

model = MLP().to(device)                               # 모델을 만들고 계산 장치로 옮긴다.
loss_fn = nn.CrossEntropyLoss()                        # 분류용 손실, softmax와 −ln(정답 확률) 평균을 한 번에 계산한다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)   # 모델의 파라미터를 lr만큼씩 고치는 도구 (모델을 옮긴 뒤에 만든다.)

# ---------- 학습 ----------
for epoch in range(1, 201):                            # 전체 데이터를 200바퀴 돈다. (1 에폭 = 3 스텝, 총 600번 갱신)
    model.train()                                      # 학습 모드로 바꾼다.
    for xb, yb in train_loader:                        # 배치(4개)를 하나씩 꺼낸다.
        xb = xb.to(device)                             # 입력 배치를 모델과 같은 장치로 옮긴다.
        yb = yb.to(device)                             # 정답 배치도 같은 장치로 옮긴다.

        optimizer.zero_grad()                          # 이전 배치의 기울기를 비운다. (비우지 않으면 누적된다.)
        pred = model(xb)                               # 예측 점수(logits)를 계산한다. (4, 2)
        loss = loss_fn(pred, yb)                       # 예측과 정답의 차이를 스칼라 손실 하나로 만든다.
        loss.backward()                                # 모든 파라미터의 기울기를 계산한다.
        optimizer.step()                               # 기울기 방향으로 파라미터를 lr만큼 고친다.

    if epoch % 50 == 0:
        print(f"epoch {epoch} loss {loss.item():.4f}")   # 50 에폭마다 마지막 배치의 손실을 출력한다.

# ---------- 평가 ----------
x_val = x_val.to(device)                               # 검증 데이터를 계산 장치로 옮긴다.
y_val = y_val.to(device)

before = {k: v.detach().clone() for k, v in model.named_parameters()}   # 평가 전 파라미터를 복사해 둔다. (비교용)

model.eval()                                           # 평가 모드로 바꾼다. (기록을 끄는 것은 아니다.)
with torch.no_grad():                                  # 이 블록 안의 계산은 기록을 남기지 않는다.
    val_logits = model(x_val)                          # 검증 데이터의 예측 점수 (4, 2)
    val_loss = loss_fn(val_logits, y_val)              # 검증 손실
    val_acc = (val_logits.argmax(dim=1) == y_val).float().mean()   # 가장 높은 점수의 class가 정답과 같은 비율

unchanged = all(torch.equal(before[k], v) for k, v in model.named_parameters())   # 평가 전후로 파라미터가 같은지 확인한다.
print("val_acc", val_acc.item())
print("unchanged", unchanged)

# ---------- 저장과 복원 ----------
torch.save(model.state_dict(), "mlp_state.pt")         # 파라미터의 이름과 값(state_dict)을 파일로 저장한다. (forward는 저장되지 않는다.)

restored = MLP().to(device)                            # 같은 구조의 새 모델을 만든다. (파라미터는 랜덤 초기값)
state = torch.load("mlp_state.pt", map_location=device, weights_only=True)   # 파일에서 저장한 값을 읽는다.
restored.load_state_dict(state)                        # 새 모델에 저장한 값을 채운다.
restored.eval()                                        # 평가 모드로 바꾼다.

with torch.no_grad():
    pred_after = restored(x_val)                       # 복원한 모델로 같은 입력을 예측한다.
print("allclose", torch.allclose(val_logits, pred_after))   # 원래 모델의 예측과 오차 범위 안에서 같은지 비교한다.
```
```
epoch 50 loss 0.xxxx
epoch 100 loss 0.xxxx
epoch 150 loss 0.xxxx
epoch 200 loss 0.xxxx
val_acc 1.0
unchanged True
allclose True
```

- 손실 값은 실행 환경과 초기값에 따라 달라지지만, 줄어드는 방향이다.

### 2. 학습 한 스텝의 흐름

```
배치를 꺼낸다 (DataLoader)
→ 장치로 옮긴다 (.to(device))
→ 기울기를 비운다 (zero_grad)
→ 예측한다 (model(xb) = forward)
→ 손실을 구한다 (CrossEntropyLoss)
→ 기울기를 계산한다 (backward)
→ 파라미터를 고친다 (step)
```

- 이 흐름을 배치마다 반복하고, 모든 배치를 한 바퀴 돌면 1 에폭이다.
- 평가는 이 흐름에서 `zero_grad`, `backward`, `step`을 빼고 `eval()`과 `no_grad`로 감싼 것이다.

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
