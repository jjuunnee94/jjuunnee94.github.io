---
title: "Object Detection SSD 04"
date: 2026-11-25 12:00:00 +0900
categories: [Codeit, 04-Study-Note]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. DataLoader 설정

이미지를 텐서로 바꾸는 변환, 한 배치로 묶는 규칙(`collate_fn`), `DataLoader`를 만들고 첫 배치를 꺼내 확인한다.

```python
from torchvision import transforms as T
from torch.utils.data import DataLoader

def get_transform():
    def transform(img, target):
        img = T.ToTensor()(img)                              # PIL 이미지 → (채널, 높이, 너비) 0~1 텐서
        return img, target
    return transform

train_dataset = PetDataset(train_xmls, image_dir, label2idx, transforms=get_transform())  # 학습 데이터셋
val_dataset = PetDataset(val_xmls, image_dir, label2idx, transforms=get_transform())      # 검증 데이터셋

def collate_fn(batch):
    return tuple(zip(*batch))                                # (이미지들), (target들)로 분리

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)   # 학습은 섞음
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)      # 검증은 안 섞음

images, targets = next(iter(train_loader))                   # 첫 배치만 꺼내 확인
print("배치 내 이미지 개수:", len(images))
print("첫 번째 이미지 텐서 모양:", images[0].shape)
print("첫 번째 정답 타겟:", targets[0])
```

```
배치 내 이미지 개수: 8
첫 번째 이미지 텐서 모양: torch.Size([3, 500, 333])
첫 번째 정답 타겟: {'boxes': tensor([[ 18.,  77., 219., 212.]]), 'labels': tensor([2])}
```

- `ToTensor()`는 (높이, 너비, 채널) 0~255 이미지를 (채널, 높이, 너비) 0~1 텐서로 바꾼다. 텐서 모양 `[3, 500, 333]`은 채널 3, 높이 500, 너비 333이다.
- 이미지마다 크기와 박스 수가 달라 기본 방식으로는 한 배치로 쌓을 수 없다. `collate_fn`이 `[(img_a, t_a), (img_b, t_b)]`를 `(img_a, img_b)`, `(t_a, t_b)` 두 묶음으로 나눈다.

## 2. 3분할로 변경

DataLoader를 만들다 보니 최종 성능을 재 볼 테스트 데이터가 없다는 것을 알게 되어, 분할 단계로 돌아가 학습/검증/테스트 3분할로 바꾼다. 처음 만든 코드는 `train_test_split(xml_files, test_size=0.2, random_state=42)`로 학습/검증 두 개만 나눴다.

```python
from sklearn.model_selection import train_test_split

train_xmls, temp_xmls = train_test_split(
    xml_files, test_size=0.30, random_state=42               # 전체의 30%를 따로 떼어 둠
)
val_xmls, test_xmls = train_test_split(
    temp_xmls, test_size=0.50, random_state=42               # 떼어 둔 것을 반씩 검증/테스트로 나눔
)

print(
    f"Train: {len(train_xmls)} | Val: {len(val_xmls)} | Test: {len(test_xmls)}"
)
```

```
Train: 2580 | Val: 553 | Test: 553
```

## 3. test 데이터셋과 로더 추가

테스트 데이터가 생겼으므로 검증과 같은 방식으로 데이터셋과 로더를 하나 더 만든다.

```python
test_dataset = PetDataset(
    test_xmls, image_dir, label2idx, transforms=get_transform()    # 테스트 데이터셋
    )
test_loader = DataLoader(
    test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn    # 테스트는 섞지 않음
    )
```

## 4. SSD300 모델 불러오기

`torchvision`의 `ssd300_vgg16`으로 모델을 만든다. 백본(VGG16)만 ImageNet 사전학습 가중치를 쓰고, 검출 헤드는 새로 학습한다. 클래스 수에는 배경을 포함한다.

```python
import torchvision
from torchvision.models.detection import ssd300_vgg16

num_classes = len(label2idx) + 1                             # 배경(0) + cat(1) + dog(2) = 3
model = ssd300_vgg16(weights=None, weights_backbone="DEFAULT", num_classes=num_classes)  # 백본만 사전학습
model.to(device)                                             # 모델을 mps로 이동
```

- 이름의 300은 입력 이미지를 모델 안에서 300×300으로 조정한다는 뜻이다.
- 학습 모드에서 `model(images, targets)`는 손실 딕셔너리를, 평가 모드에서 `model(images)`는 이미지별 예측(`boxes`, `labels`, `scores`)을 돌려준다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
