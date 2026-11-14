---
title: "CNN - 10. evaluation"
date: 2026-11-14 09:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, cnn, 평가, argmax, accuracy, eval, no_grad, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---

## 1. eval()과 no_grad()

- 평가 : 학습된 모델이 처음 보는 데이터(test)를 얼마나 맞히는지 측정
- 가중치를 바꾸지 않으므로 `zero_grad`, `backward`, `step`이 없음

| | 하는 일 | 필요한 이유 |
|:---:|:---|:---|
| `model.eval()` | 학습과 평가에서 동작이 다른 층을 평가 방식으로 전환 | Dropout은 전부 사용, BN은 저장된 값 사용 |
| `torch.no_grad()` | backward에 쓸 연산 기록을 만들지 않음 | 기록과 메모리 비용 절약 |

- 이 SimpleCNN에는 Dropout, BN이 없지만 평가 의도를 분명히 하려고 사용
- `eval()`만으로는 미분 기록이 꺼지지 않음, `no_grad()`만으로는 평가 모드가 되지 않음

## 2. argmax

- logits `[N,10]` → 이미지마다 최대 점수의 **클래스 번호(위치)** 선택 → `pred` `[N]`
- `dim=1` : 클래스 축에서 고름 (행마다 하나), `dim=0`은 이미지끼리 비교하므로 사용하지 않음

📍 **[예시]** 이미지 3장, 클래스 4개

| | 클래스 0 | 클래스 1 | 클래스 2 | 클래스 3 | 최댓값 위치 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 이미지 1 | 0.5 | −1.2 | **3.1** | 2.9 | 2 |
| 이미지 2 | 1.0 | **2.0** | −0.5 | 0.3 | 1 |
| 이미지 3 | −2.0 | −1.0 | −3.0 | **−0.5** | 3 |

## 3. correct / total

- `pred == labels` : 같은 이미지끼리 비교한 True/False
- `.sum()` : True 개수, `.item()` : Python 숫자로 꺼냄

$$
\text{accuracy} = \frac{\text{correct}}{\text{total}}
$$

📍 **[예시]** `pred = [2, 1, 3]`, `labels = [2, 0, 3]`

| 이미지 | pred | labels | 같은가 | 숫자로 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | 2 | True | 1 |
| 2 | 1 | 0 | False | 0 |
| 3 | 3 | 3 | True | 1 |

- correct = 2, total = 3 → accuracy $= 2/3 \approx 0.667$
- 분자와 분모는 같은 테스트 데이터에서 셈 (total이 테스트 전체 이미지 수와 같은지 확인)

## 4. loss, accuracy, 개별 예측

| 질문 | 보는 값 | 대상 |
|:---|:---|:---|
| 학습 중 목적값은 어느 정도였나? | 훈련 평균 loss | 훈련 데이터, 여러 갱신 시점의 누적 |
| 테스트 전체 중 얼마나 맞혔나? | test accuracy | 테스트 전체 |
| 이 이미지들은 맞혔나? | 개별 pred와 true | 일부 이미지 몇 장 |

- 몇 장의 예측이 전체 정확도를 대표하지는 않음
- 수치 하나만 보지 않고 어떤 데이터, 어떤 모집단, 어느 시점에서 얻었는지 함께 확인

## 5. 💻 [코드]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/cnn_practice.ipynb)

```python
import torch

logits = torch.tensor([[0.5, -1.2, 3.1, 2.9],
                       [1.0, 2.0, -0.5, 0.3],
                       [-2.0, -1.0, -3.0, -0.5]])   # 이미지 3장, 클래스 4개 점수
labels = torch.tensor([2, 0, 3])                     # 정답

pred = logits.argmax(dim=1)                          # 이미지마다 최대 점수의 클래스 번호
print(pred)
print(pred == labels)                                # 같은 이미지끼리 비교
print((pred == labels).sum().item())                 # 맞힌 개수
print((pred == labels).sum().item() / labels.size(0))  # 정확도
print(logits.argmax(dim=0))                          # dim=0 : 클래스마다 이미지끼리 비교 (사용 X)

w = torch.ones(1, requires_grad=True)                # 미분 기록 대상
a = w * 2                                            # 기록을 만드는 계산
with torch.no_grad():
    b = w * 2                                        # 기록을 만들지 않는 계산
print(a.requires_grad, b.requires_grad)
```

```
tensor([2, 1, 3])
tensor([ True, False,  True])
2
0.6666666666666666
tensor([1, 1, 0, 0])
True False
```

```python
model.eval()                                       # 평가 모드
correct, total = 0, 0                              # 맞힌 수, 본 이미지 수
with torch.no_grad():                              # 미분 기록 없이 계산
    for images, labels in test_loader:             # 테스트 배치를 하나씩
        logits = model(images)                     # 점수 계산 [N,10]
        pred = logits.argmax(dim=1)                # 이미지마다 최대 점수 클래스 [N]
        correct += (pred == labels).sum().item()  # 맞힌 개수 누적
        total += labels.size(0)                    # 이번 배치 이미지 수 누적
print(correct / total)                             # 정확도
```

```
model과 test_loader가 앞에서 정의되어 있어야 해서 출력 블록은 없는 코드예시.
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
