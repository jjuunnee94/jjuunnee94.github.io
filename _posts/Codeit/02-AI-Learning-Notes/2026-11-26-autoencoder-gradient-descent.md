---
title: "AutoEncoder - 03. 경사 하강법과 파라미터 갱신"
date: 2026-11-26 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, gradient-descent, learning-rate, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 경사 하강법
- 한 바퀴: 순전파 → 손실 → 역전파 → 모든 가중치 한꺼번에 갱신
- 기울기가 음수면 가중치 증가, 양수면 감소
- 학습률(η): 이동량의 크기 조절, 크면 지나칠 수 있음

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 가중치 갱신 | 새 가중치 = 현재 가중치 − 학습률 × 기울기 | $\theta_{new}=\theta-\eta\frac{\partial L}{\partial\theta}$ |

📍[예시] 앞 편 기울기 사용, $\eta=0.1$

| 가중치 | 현재 값 | 이동량(η×기울기) | 갱신 값 |
|:---:|:---:|:---:|:---:|
| $e_1$ | 0.5 | 0.1×(−3) = −0.3 | 0.5−(−0.3) = **0.8** |
| $e_2$ | 0.2 | 0.1×(−4.5) = −0.45 | 0.2−(−0.45) = **0.65** |
| $d_1$ | 1.0 | 0.1×(−0.64) = −0.064 | 1.0−(−0.064) = **1.064** |
| $d_2$ | 0.5 | 0.1×(−3.52) = −0.352 | 0.5−(−0.352) = **0.852** |

## 2. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
x = np.array([2.0, 3.0])  # 입력(x)
e = np.array([0.5, 0.2])  # 인코더 가중치(e)
d = np.array([1.0, 0.5])  # 디코더 가중치(d)
eta = 0.1  # 학습률(η)
n = x.size  # 입력 칸 수(n)
z = (e * x).sum()  # 순전파: 잠재 표현(z)
rec = d * z  # 복원값(x̂)
grad_rec = (2 / n) * (rec - x)  # 출력 기울기
grad_d = grad_rec * z  # 디코더 가중치 기울기
grad_z = (grad_rec * d).sum()  # 잠재 표현 기울기
grad_e = grad_z * x  # 인코더 가중치 기울기
e = e - eta * grad_e  # 인코더 가중치 갱신
d = d - eta * grad_d  # 디코더 가중치 갱신
print(e.round(3), d.round(3))  # 갱신된 가중치 출력
```

```
[0.8  0.65] [1.064 0.852]
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
