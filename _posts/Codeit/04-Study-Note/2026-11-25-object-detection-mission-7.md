---
title: "Object Detection SSD 07"
date: 2026-11-25 15:00:00 +0900
categories: [Codeit, 04-Study-Note]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 이상한 값 확인

학습 로그를 다시 보니 값이 이상했다.

```
Epoch 1/5 [Train]:   0%|          | 0/461 [00:00<?, ?it/s]
Epoch 1/5 [Val]:   0%|          | 0/461 [00:00<?, ?it/s]
Epoch [1/5] Train Loss: 1.2789 | Val mAP: 0.6855 | mAP50: 0.9919 <-- BEST saved
```

- 검출 헤드를 새로 학습하는 모델이 1에폭부터 mAP50 0.99를 기록했다.
- 학습과 검증의 배치 수가 둘 다 461이다. 데이터를 나눴다면 학습과 검증의 배치 수는 달라야 한다.
- 3,686장을 배치 크기 8로 나누면 460.75 → 461배치, 즉 둘 다 이미지 전체를 쓰고 있다는 뜻이다.

## 2. 원인 찾기: PetDataset의 `__init__`

분할한 목록(`xml_files`)을 받아 저장한 바로 아래에서, `xml_dir` 안의 XML 전체를 다시 목록으로 만들어 덮어쓰고 있었다. 그래서 학습·검증·테스트 데이터셋이 모두 3,686개 전체를 사용했다. 학습에 쓴 이미지로 검증과 평가까지 하는 데이터 누수다. 덮어쓰는 블록을 지워 받은 목록을 그대로 쓰도록 고친다.

```python
class PetDataset(Dataset):
    def __init__(self, xml_files, image_dir, label2idx, transforms=None):
        self.xml_files = xml_files                           # 받은 분할 목록을 그대로 사용
        self.image_dir = image_dir
        self.label2idx = label2idx
        self.transforms = transforms

        # [데이터 누수 원인 - 수정 전 코드]
        # 받은 xml_files(분할된 목록)를 아래 블록이 xml_dir 전체 목록(3,686개)으로 덮어씀
        #
        # self.xml_files = sorted([
        #     os.path.join(xml_dir, f)
        #     for f in os.listdir(xml_dir)
        #     if f.endswith(".xml")
        # ])
        #
        # [수정] 위 블록을 삭제
```

- `__len__`과 `__getitem__`은 3편과 같다.

## 3. 수정 확인

데이터셋 클래스와 데이터셋을 만드는 셀을 다시 실행하고, 각 데이터셋의 크기를 확인한다.

```python
print(len(train_dataset), len(val_dataset), len(test_dataset))     # 데이터셋 크기 확인
```

```
2580 553 553
```

- 분할 크기(2,580 / 553 / 553)와 같다.
- 지금까지 학습한 모델은 전체 이미지를 본 상태라서, 모델을 불러오는 셀부터 다시 실행해 새 모델로 처음부터 학습한다.

## 4. 10에폭 재학습

학습 코드는 5편과 같고 `num_epochs = 10`으로만 바꿨다. 학습 장치는 다시 VSCode MPS로 돌아왔다.

```
Epoch 1/10 [Train]: 100%|██████████| 323/323 [02:23<00:00,  2.25it/s]
Epoch 1/10 [Val]: 100%|██████████| 70/70 [00:14<00:00,  4.89it/s]
Epoch [1/10] Train Loss: 6.8449 | Val mAP: 0.0326 | mAP50: 0.1372 <-- BEST saved
Epoch [2/10] Train Loss: 4.5404 | Val mAP: 0.1684 | mAP50: 0.4786 <-- BEST saved
Epoch [3/10] Train Loss: 3.5795 | Val mAP: 0.3333 | mAP50: 0.7875 <-- BEST saved
Epoch [4/10] Train Loss: 2.6554 | Val mAP: 0.4615 | mAP50: 0.9203 <-- BEST saved
Epoch [5/10] Train Loss: 1.9464 | Val mAP: 0.5358 | mAP50: 0.9518 <-- BEST saved
Epoch [6/10] Train Loss: 1.4792 | Val mAP: 0.5266 | mAP50: 0.9566
Epoch [7/10] Train Loss: 1.1504 | Val mAP: 0.5764 | mAP50: 0.9682 <-- BEST saved
Epoch [8/10] Train Loss: 0.9842 | Val mAP: 0.5710 | mAP50: 0.9629
Epoch [9/10] Train Loss: 0.8908 | Val mAP: 0.5507 | mAP50: 0.9541
Epoch [10/10] Train Loss: 0.7687 | Val mAP: 0.5610 | mAP50: 0.9613
```

- 배치 수가 학습 323, 검증 70으로 나뉘었다. 2,580장과 553장을 8로 나눈 값(322.5 → 323, 69.1 → 70)과 같다.
- 에폭당 약 2분 40초(MPS)가 걸렸다. Colab T4는 약 2분 20초였다.
- Train Loss는 6.8449 → 0.7687로 줄었고, 검증 mAP는 7에폭(0.5764)에서 가장 높아 그 가중치가 best로 저장됐다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
