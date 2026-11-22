---
title: "AutoEncoder 기호 및 공식 정리 4"
date: 2026-11-22 17:30:00 +0900
categories: [Codeit, 04-Study-Note]
tags: [codeit, autoencoder, symbol, formula, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

#### 🔍[기호] 학습·Sparse AE·VAE 기호 · 이름 · 뜻

| 기호 | 이름 | 뜻 |
|:---:|:---:|:---:|
| **$N$, $B$** | **훈련 장 수, 배치 크기** | 훈련 이미지 수, 한 번에 처리하는 장 수 |
| **$\lambda$** | **희소 규제 가중치** | 희소성 비용의 비중 |
| **$\mu$, $\sigma$** | **평균, 표준편차** | 입력별 잠재분포를 정하는 값 |
| **$\varepsilon$** | **표준정규 난수** | 외부에서 뽑는 무작위 값 |
| **$q$** | **입력별 분포** | 인코더가 정한 잠재분포 |
| **$\text{KL}$** | **KL 비용** | $q$와 prior($N(0, I)$)의 차이 |

#### 📍[공식] 학습 횟수

- **배치 수**
  - 배치 수 = 올림(훈련 장 수 ÷ 배치 크기)

$$ \left\lceil N / B \right\rceil $$

- **갱신 횟수**
  - 갱신 횟수 = 배치 수 × 에폭 수

#### 📍[공식] 평가

- **활성 비율**
  - 활성 비율 = 잠재 표현이 0이 아닌 값의 비율
- **개선 판단**
  - 모델 오차 < 기준선 오차(noisy ↔ clean)

#### 📍[공식] Sparse AE

- **손실** (잠재 표현이 0에 가깝도록)
  - 손실 = 평균제곱오차 + 희소 규제 가중치 × 잠재 표현 절댓값의 평균

$$ L = \text{MSE}(\hat{x}, x) + \lambda\,\text{mean}(|z|) $$

#### 📍[공식] VAE

- **재매개변수화**
  - 잠재 표현 = 평균 + 표준편차 × 표준정규 난수

$$ z = \mu + \sigma \odot \varepsilon, \quad \varepsilon \sim N(0, I) $$

- **손실**
  - 손실 = 재구성 비용 + KL 비용

$$ L = L_{\text{recon}} + \text{KL}\big(q(z|x)\ \|\ N(0, I)\big) $$

- **KL 비용** (분포가 가우시안일 때, 슬라이드에 없음)
  - KL = −½ × (1 + log 분산 − 평균² − 분산)의 합

$$ \text{KL}=-\frac{1}{2}\sum\left(1+\log\sigma^2-\mu^2-\sigma^2\right) $$

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
