---
title: "Object Detection SSD 05"
date: 2026-11-25 13:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 모델 학습 및 검증

에폭마다 학습 → 검증 → 검증 mAP 계산을 반복하고, 검증 mAP가 가장 높을 때만 가중치를 저장한다. VSCode(MPS)에서 에폭당 약 6분이 걸려 에폭 수를 5로 정했다.

```python
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm

BEST_PATH = "best_ssd300_vgg16.pth"                          # best 가중치 저장 경로
best_map = float("-inf")                                     # 지금까지 최고 mAP
history = []                                                 # 에폭별 기록

metric = MeanAveragePrecision()                              # mAP 계산기
num_epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
  model.train()                                              # 학습 모드
  train_loss = 0.0

  for images, targets in tqdm(
      train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
  ):
    images = [img.to(device) for img in images]              # 이미지를 mps로 이동
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)                       # 학습 모드: 손실 딕셔너리 반환
    losses = sum(loss for loss in loss_dict.values())        # 손실 합산

    optimizer.zero_grad()                                    # 기울기 초기화
    losses.backward()                                        # 역전파
    optimizer.step()                                         # 가중치 갱신

    train_loss += losses.item()

  avg_train_loss = train_loss / len(train_loader)            # 에폭 평균 손실

  model.eval()                                               # 평가 모드
  with torch.no_grad():                                      # 기울기 계산 끔
    for images, targets in tqdm(
        val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"
    ):
      images = [img.to(device) for img in images]
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

      predictions = model(images)                            # 평가 모드: 예측 반환
      metric.update(predictions, targets)                    # mAP 계산용으로 쌓기

  val_metrics = metric.compute()                             # 쌓은 결과로 mAP 계산
  val_map = val_metrics["map"].item()                        # mAP(IoU 0.5~0.95)
  val_map50 = val_metrics["map_50"].item()                   # mAP50(IoU 0.5)
  metric.reset()                                             # 다음 에폭을 위해 초기화

  history.append({
      "epoch": epoch + 1,
      "loss": avg_train_loss,
      "val_map": val_map,
      "val_map50": val_map50,
  })

  if val_map > best_map:                                     # 최고 기록이면
    best_map = val_map
    torch.save(model.state_dict(), BEST_PATH)                # 가중치 저장
    best_mark = " <-- BEST saved"
  else:
    best_mark = ""

  print(
      f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} |"
      f" Val mAP: {val_map:.4f} | mAP50: {val_map50:.4f}{best_mark}"
  )
```

```
Epoch [1/5] Train Loss: 1.2789 | Val mAP: 0.6855 | mAP50: 0.9919 <-- BEST saved
Epoch [2/5] Train Loss: 1.0624 | Val mAP: 0.7165 | mAP50: 0.9925 <-- BEST saved
Epoch [3/5] Train Loss: 0.8462 | Val mAP: 0.7232 | mAP50: 0.9991 <-- BEST saved
Epoch [4/5] Train Loss: 0.7096 | Val mAP: 0.7599 | mAP50: 0.9995 <-- BEST saved
Epoch [5/5] Train Loss: 0.6216 | Val mAP: 0.7311 | mAP50: 0.9949
```

- 한 스텝은 손실 합산 → `zero_grad` → `backward` → `step` 순서다. 옵티마이저는 AdamW, 학습률은 0.0001이다.
- 검증 mAP는 배치마다 `update`로 쌓고, 에폭 끝에 `compute`로 계산한 뒤 `reset`한다. `mAP`는 IoU 0.5~0.95 평균, `mAP50`은 IoU 0.5 기준 값이다.
- Train Loss는 1.2789 → 0.6216으로 줄었고, 검증 mAP가 가장 높은 4에폭(0.7599)의 가중치가 `best_ssd300_vgg16.pth`로 저장됐다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
