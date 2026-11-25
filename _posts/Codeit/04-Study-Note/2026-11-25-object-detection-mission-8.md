---
title: "Object Detection SSD 08"
date: 2026-11-25 16:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 학습 곡선과 best 에폭 찾기

에폭별 학습 손실과 검증 mAP, mAP50을 그래프로 그리고, 검증 mAP가 가장 높은 에폭을 찾는다. 수업에서 쓴 코드를 그대로 사용했다.

```python
ep = [h["epoch"] for h in history]                           # 에폭 번호 목록
fig, axes = plt.subplots(1, 2, figsize=(12, 4))              # 그래프 2개를 나란히

axes[0].plot(ep, [h["loss"] for h in history], marker="o"); axes[0].set_title("train loss")   # 학습 손실

axes[1].plot(ep, [h["val_map"] for h in history], marker="o", label="mAP50-95")               # 검증 mAP
axes[1].plot(ep, [h["val_map50"] for h in history], marker="o", label="mAP50"); axes[1].legend(); axes[1].set_title("val")   # 검증 mAP50
plt.show()

best_h = max(history, key=lambda h: h["val_map"])            # 검증 mAP가 가장 높은 에폭
print("best epoch:", best_h["epoch"], best_h)
```

```
best epoch: 7 {'epoch': 7, 'loss': 1.1504454012994796, 'val_map': 0.5763536095619202, 'val_map50': 0.9682425260543823}
```

- `history`에는 에폭마다 `loss`, `val_map`, `val_map50`이 담겨 있다.
- best 에폭은 7이고, 이때 저장한 가중치가 `best_ssd300_vgg16.pth`다.

## 2. test 평가 — mAP

6편의 평가 코드를 그대로 다시 실행했다. best 가중치를 불러와 test 데이터 553장으로 mAP를 계산한다.

```
test mAP (IoU 0.5:0.95) = 0.565
test mAP50              = 0.959
test mAP75              = 0.589
  cat     AP = 0.589
  dog     AP = 0.541
```

- IoU 기준이 0.5에서 0.75로 엄격해지면 점수가 0.959에서 0.589로 내려간다.
- 클래스별 AP는 cat 0.589, dog 0.541이다.
- best 에폭의 검증 mAP(0.5764)와 test mAP(0.565)가 비슷하다.

## 3. MPS 메모리 오류 해결

6편에서 MPS 메모리 초과가 났던 평가가 같은 MPS 환경에서 오류 없이 끝났다.

- 데이터 누수가 있을 때는 test 평가 대상이 3,686장이었고, 수정 후에는 553장으로 줄었다.
- 메모리 오류의 원인은 이렇게 평가 대상이 컸던 것으로 보인다. 메모리 사용량을 직접 비교하지는 않았다.
- 준비해 둔 CPU 전환 코드는 쓰지 않아도 됐다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
