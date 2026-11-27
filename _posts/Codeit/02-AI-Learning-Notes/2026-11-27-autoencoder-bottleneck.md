---
title: "AutoEncoder - 04. 병목과 정보 손실"
date: 2026-11-27 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, bottleneck, information-loss, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 병목(bottleneck)
- 잠재 표현(z)의 칸 수가 입력보다 적은 구조
- 정보 일부를 복원할 수 없음 = 정보 손실 
- 입력을 그대로 베끼지 못해 복원에 필요한 정보만 남기도록 학습
- 잠재 차원이 입력 이상이면 통째로 통과해 압축 의미 없음

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 병목 조건 | 잠재 차원(k) < 입력 차원(m) | $k < m$ |
| 압축 비율 | 입력 차원(m) ÷ 잠재 차원(k) | $m / k$ |

📍[예시] 4→2→4, 규칙: 앞 두 칸 평균·뒤 두 칸 평균 → 각 칸 두 번 복사 (압축 비율 4÷2=**2**)

| 입력(x) | 잠재 표현(z) | 복원값(x̂) | 차이(x̂−x) | 제곱 | 평균제곱오차(L) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| (2, 6, 5, 1) | $(\frac{2+6}{2},\ \frac{5+1}{2})$ = **(4, 3)** | **(4, 4, 3, 3)** | (2, −2, −2, 2) | (4, 4, 4, 4) | $\frac{16}{4}$ = **4** |

- 입력 (4, 4, 3, 3)도 같은 z (4, 3) → 복원 구분 불가, 이때 $L=0$
- 칸 안의 값 차이가 정보 손실, 평균제곱오차로 나타남

## 2. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
x = np.array([2.0, 6.0, 5.0, 1.0])  # 입력(x) 네 칸
z = np.array([(x[0] + x[1]) / 2, (x[2] + x[3]) / 2])  # 잠재 표현(z): 앞 두 칸 평균, 뒤 두 칸 평균
rec = np.array([z[0], z[0], z[1], z[1]])  # 복원값(x̂): z의 각 칸을 두 번씩 복사
mse = ((rec - x) ** 2).mean()  # 평균제곱오차: 차이 제곱의 평균
print(z, rec, mse)  # 결과 출력
```

```
[4. 3.] [4. 4. 3. 3.] 4.0
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
