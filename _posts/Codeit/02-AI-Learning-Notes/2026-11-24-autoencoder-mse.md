---
title: "AutoEncoder - 04. 병목과 정보 손실"
date: 2026-11-27 09:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, autoencoder, bottleneck, nn-linear, pytorch, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 병목(bottleneck)
- 잠재 표현(z)의 칸 수가 입력보다 적은 구조, 가중치 값이 아니라 모양(4→2)에서 생김
- 정보 일부를 복원할 수 없음 = 정보 손실
- 가중치는 무작위로 시작해 학습으로 정해짐
- 파라미터 수: `Linear(4, 2)` 4×2+2 = **10**, `Linear(2, 4)` 2×4+4 = **12**

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 병목 조건 | 잠재 차원(k) < 입력 차원(m) | $k < m$ |
| 압축 비율 | 입력 차원(m) ÷ 잠재 차원(k) | $m / k$ |
| 인코더 | z = ReLU(인코더 가중치 × 입력 + 편향) | $z=\text{ReLU}(W_e x+b_e)$ |
| 디코더 | x̂ = 디코더 가중치 × z + 편향 | $\hat{x}=W_d z+b_d$ |

📍[예시] $W_e$=[[0.5,0.5,0,0],[0,0,0.5,0.5]], $b_e$=(0,−4), $W_d$=[[1,0],[1,0],[0,1],[0,1]], $b_d$=0, $x=(2,6,5,1)$

| 인코더 선형 | ReLU 후 z | 복원값(x̂) | 평균제곱오차(L) |
|:---:|:---:|:---:|:---:|
| $(0.5\cdot2+0.5\cdot6,\ 0.5\cdot5+0.5\cdot1-4)$ = (4, −1) | **(4, 0)** | **(4, 4, 0, 0)** | $\frac{4+4+25+1}{4}$ = **8.5** |

## 2. 💻 [코드]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjuunnee94/jjuunnee94.github.io/blob/main/notebooks/AutoEncoder_practice.ipynb)

```python
import torch  # 파이토치
import torch.nn as nn  # 신경망 층 모음

class Autoencoder(nn.Module):  # 모델의 틀, nn.Module 기능을 물려받음
    def __init__(self):  # 모델을 만들 때 층을 준비
        super().__init__()  # 물려받은 틀의 준비 작업 실행
        self.encoder = nn.Sequential(  # 인코더: 층을 순서대로 연결
            nn.Linear(4, 2),  # 4칸 → 2칸(병목), 가중치·편향 학습
            nn.ReLU()  # 음수는 0
        )
        self.decoder = nn.Sequential(  # 디코더
            nn.Linear(2, 4)  # 2칸 → 4칸 복원
        )

    def forward(self, x):  # 순전파 계산 순서
        z = self.encoder(x)  # 잠재 표현(z)
        rec = self.decoder(z)  # 복원값(x̂)
        return rec  # 복원값 반환

model = Autoencoder()  # 모델 생성
x = torch.tensor([[2.0, 6.0, 5.0, 1.0]])  # 입력(x), 모양 (1, 4)
z = model.encoder(x)  # 인코더만 통과
rec = model(x)  # 전체 통과
print(z.shape, rec.shape)  # 모양 출력
```

```
torch.Size([1, 2]) torch.Size([1, 4])
```

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
