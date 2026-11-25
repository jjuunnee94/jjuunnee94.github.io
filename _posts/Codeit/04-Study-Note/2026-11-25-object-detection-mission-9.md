---
title: "Object Detection SSD 09"
date: 2026-11-25 17:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. test 평가 — IoU

```python
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import box_iou


def best_ious(preds, targets, score_thr=0.5):
  rows = []
  for idx, (p, t) in enumerate(zip(preds, targets)):         # 이미지마다 예측 p, 정답 t
    keep = p["scores"] >= score_thr                          # 신뢰도가 임계값 이상인 예측만 남김
    pb, pl = p["boxes"][keep], p["labels"][keep]             # 남긴 예측의 박스, 라벨
    for gb, gl in zip(t["boxes"], t["labels"]):              # 정답 박스마다
      same = pl == gl                                        # 같은 클래스인 예측
      iou = (
          box_iou(gb[None], pb[same]).max().item() if same.any() else 0.0   # 가장 큰 IoU, 같은 클래스 예측이 없으면 0
      )
      rows.append(
          {"image": idx, "label": idx2label[int(gl)], "iou": iou}   # 이미지 번호, 클래스, IoU 기록
      )
  return rows


iou_rows = best_ious(test_preds, test_targets)               # test 예측과 정답으로 계산
ious = np.array([r["iou"] for r in iou_rows])                # IoU만 모아 배열로

print(f"정답 {len(ious)}개, 평균 IoU = {ious.mean():.3f}")
print(
    f"찾은 것(IoU>0)만의 평균 IoU ="
    f" {(ious[ious > 0].mean() if (ious > 0).any() else 0.0):.3f}, 못 찾은"
    f" 정답 = {(ious == 0).sum()}개"
)

plt.figure(figsize=(6, 3.5))
plt.hist(ious, bins=20, range=(0, 1))                        # IoU 분포 히스토그램
plt.axvline(0.5, color="red", ls="--", label="IoU 0.5")      # IoU 0.5 기준선
plt.xlabel("정답별 최고 IoU")
plt.ylabel("개수")
plt.legend()
plt.show()
```

```
정답 554개, 평균 IoU = 0.746
찾은 것(IoU>0)만의 평균 IoU = 0.789, 못 찾은 정답 = 30개
```

- 정답 554개는 이미지 553장보다 하나 많다. 객체가 둘인 이미지가 있다는 뜻이다.
- 평균 IoU 0.746에는 못 찾은 정답(IoU 0)도 들어간다. 찾은 것만의 평균은 0.789다.
- 못 찾은 정답 30개는 신뢰도 0.5 이상인 같은 클래스 예측이 없거나, 예측이 하나도 겹치지 않은 정답이다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
