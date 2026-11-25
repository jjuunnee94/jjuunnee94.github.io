---
title: "Object Detection SSD 06"
date: 2026-11-25 14:00:00 +0900
categories: [Codeit, 02-AI-Learning-Notes]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. test 평가 — mAP

저장해 둔 best 가중치를 불러와 test 데이터로 mAP를 계산한다. 예측과 정답을 모으는 `predict`, mAP를 계산하는 `compute_map`을 함수로 만든다.

```python
def predict(model, loader):
  model.eval()                                               # 평가 모드
  preds, targets = [], []
  with torch.no_grad():                                      # 기울기 계산 끔
    for images, tgts in loader:
      out = model([im.to(device) for im in images])          # 이미지를 device로 보내 예측
      preds += [{k: v.cpu() for k, v in o.items()} for o in out]       # 예측 결과를 cpu로 옮겨 저장
      targets += [{k: v.cpu() for k, v in t.items()} for t in tgts]    # 정답도 cpu로 옮겨 저장
  return preds, targets

def compute_map(preds, targets, **kw):
  m = MeanAveragePrecision(**kw)                             # mAP 계산기
  m.update(preds, targets)                                   # 예측·정답 넣기
  return m.compute()                                         # mAP 계산

model = model.to(device)
model.load_state_dict(torch.load(BEST_PATH, map_location=device))    # 저장해 둔 best 가중치 불러오기
test_preds, test_targets = predict(model, test_loader)               # test 예측
test_res = compute_map(test_preds, test_targets, class_metrics=True) # 클래스별 AP 포함 계산

print(f"test mAP (IoU 0.5:0.95) = {test_res['map']:.3f}")
print(f"test mAP50              = {test_res['map_50']:.3f}")
print(f"test mAP75              = {test_res['map_75']:.3f}")
for c, ap in zip(
    test_res["classes"].tolist(), test_res["map_per_class"].tolist()
):
  print(f"  {idx2label[c]:7s} AP = {ap:.3f}")                     # 클래스별 AP
```

```
RuntimeError: MPS backend out of memory (MPS allocated: 42.44 GiB, other allocations: 5.73 MiB, max allowed: 42.43 GiB). Tried to allocate 2.00 KiB on shared pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).
```

- 평가를 시작하자 MPS 메모리 할당 한도(42.43 GiB)를 넘어 `RuntimeError`가 났다.
- 오류 메시지는 한도를 해제하는 환경 변수(`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`)를 안내하지만, 시스템 오류가 날 수 있다고 함께 경고한다.
- `mAP`는 IoU 0.5~0.95 평균, `mAP50`·`mAP75`는 IoU 0.5·0.75 기준 값이고, `class_metrics=True`로 클래스별 AP도 구한다.

## 2. CPU 전환 코드 준비

메모리 부족을 피하려고 평가를 CPU에서 하도록 바꾼 코드를 만들었다. 모델과 이미지를 보내는 장치만 `eval_device`로 바꾼 것이고, 실행하지는 않았다. CPU는 너무 느렸기 때문이다.

```python
eval_device = torch.device("cpu")                            # 평가에 쓸 장치를 cpu로

def predict(model, loader):
  model.eval()
  preds, targets = [], []
  with torch.no_grad():
    for images, tgts in loader:
      out = model([im.to(eval_device) for im in images])     # 이미지를 cpu로 보내 예측
      preds += [{k: v.cpu() for k, v in o.items()} for o in out]
      targets += [{k: v.cpu() for k, v in t.items()} for t in tgts]
  return preds, targets

def compute_map(preds, targets, **kw):
  m = MeanAveragePrecision(**kw)
  m.update(preds, targets)
  return m.compute()

model = model.to(eval_device)                                # 모델을 cpu로 이동
model.load_state_dict(torch.load(BEST_PATH, map_location=eval_device))    # 가중치도 cpu로 불러오기
test_preds, test_targets = predict(model, test_loader)
test_res = compute_map(test_preds, test_targets, class_metrics=True)

print(f"test mAP (IoU 0.5:0.95) = {test_res['map']:.3f}")
print(f"test mAP50              = {test_res['map_50']:.3f}")
print(f"test mAP75              = {test_res['map_75']:.3f}")
for c, ap in zip(
    test_res["classes"].tolist(), test_res["map_per_class"].tolist()
):
  print(f"  {idx2label[c]:7s} AP = {ap:.3f}")
```

## 3. Google Colab에서 실행

CPU로 평가하는 대신 같은 코드를 Google Colab의 T4 GPU에서 실행했다.

- Colab T4 GPU는 1에폭에 약 4분이 걸려, VSCode MPS(약 6분)보다 빨랐다.
- 다만 T4 GPU 연결이 불안정했다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
