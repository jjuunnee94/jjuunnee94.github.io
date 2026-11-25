---
title: "Object Detection SSD 03"
date: 2026-11-25 11:00:00 +0900
categories: [Codeit, 04-Study-Note]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 커스텀 Dataset 정의

XML 어노테이션과 이미지를 읽어 `(이미지, target)` 한 쌍을 돌려주는 `PetDataset`을 만든다. `Dataset`을 상속해 `__init__`, `__len__`, `__getitem__` 세 메서드를 정의한다. target은 박스 좌표(`boxes`)와 클래스 번호(`labels`)를 담은 딕셔너리다.

```python
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset


class PetDataset(Dataset):
    def __init__(self, xml_files, image_dir, label2idx, transforms=None):
        self.xml_files = xml_files                           # 받은 XML 목록
        self.image_dir = image_dir                           # 이미지 폴더
        self.label2idx = label2idx                           # 클래스 이름 → 번호
        self.transforms = transforms                         # 이미지 변환 함수
        self.xml_files = sorted([                            # xml_dir 안의 XML 경로를 정렬해 저장
            os.path.join(xml_dir, f)
            for f in os.listdir(xml_dir)
            if f.endswith(".xml")
        ])

    def __len__(self):
        return len(self.xml_files)                           # 데이터 개수

    def __getitem__(self, idx):
        xml_path = self.xml_files[idx]                       # idx번째 XML
        root = ET.parse(xml_path).getroot()                  # XML 읽기
        img_name = root.find("filename").text                # 이미지 파일명
        img_path = os.path.join(self.image_dir, img_name)    # 이미지 경로
        img = Image.open(img_path).convert("RGB")            # 이미지 열기(3채널로 통일)

        boxes = []
        labels = []

        for obj in root.findall("object"):                   # 객체마다 반복
            class_name = obj.find("name").text
            labels.append(self.label2idx[class_name])        # 클래스 번호

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])           # 박스 좌표

        boxes = torch.tensor(boxes, dtype=torch.float32)     # 박스 텐서
        labels = torch.tensor(labels, dtype=torch.int64)     # 라벨 텐서

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)       # 이미지를 텐서로 변환

        return img, target
```

- `__init__`은 경로와 변환 함수를 저장하고, `__len__`은 데이터 개수를, `__getitem__`은 번호 하나를 받아 `(이미지, target)`을 돌려준다.
- `__getitem__`은 XML 파싱 → 이미지 열기 → 박스·라벨 모으기 → 텐서 변환 → 반환 순서로 동작한다.
- `boxes`는 `float32`로 `[xmin, ymin, xmax, ymax]` 목록이고, `labels`는 `int64`로 `cat`은 1, `dog`는 2다.
- 이미지마다 객체 수가 달라 박스 개수가 제각각이라, target을 이미지별 딕셔너리로 둔다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
