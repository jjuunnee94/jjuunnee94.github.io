---
title: "Object Detection SSD 01"
date: 2026-11-25 09:00:00 +0900
categories: [Codeit, 04-Study-Note]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 데이터셋 내려받기

Oxford-IIIT Pet 데이터셋(고양이·개 사진과 XML 어노테이션)을 `kagglehub`로 내려받는다.

```python
import kagglehub

path = kagglehub.dataset_download("zippyz/cats-and-dogs-breeds-classification-oxford-dataset")  # 데이터셋 내려받기
print("Path to dataset files:", path)                      # 저장된 경로 출력
```

```
Path to dataset files: /Users/.../.cache/kagglehub/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset/versions/1
```

## 2. 장치(device) 설정

학습에 쓸 장치를 자동으로 고른다. 애플 GPU(`mps`) → 엔비디아 GPU(`cuda`) → `cpu` 순서로 확인한다.

```python
import torch

device = torch.device("mps"                                # 애플 GPU가 있으면 mps
                      if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()   # 없으면 엔비디아 GPU
                      else "cpu")                          # 둘 다 없으면 cpu
print(device)
```

```
mps
```

## 3. 데이터 경로 설정

내려받은 폴더 안에서 이미지 폴더와 XML 어노테이션 폴더의 경로를 변수에 저장한다.

```python
import os

image_dir = os.path.join(path, "images", "images")                    # 이미지 폴더
xml_dir = os.path.join(path, "annotations", "annotations", "xmls")    # XML 어노테이션 폴더
```

## 4. 데이터 개수 확인

XML 파일 경로를 모두 모아 개수를 센다.

```python
import glob

xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))     # xml_dir 안의 .xml 파일 경로 목록
print(len(xml_files))                                      # 파일 개수
```

```
3686
```

- 이미지 한 장마다 XML 어노테이션이 하나씩 있고, XML 안에 대응하는 이미지 이름(`filename`)이 적혀 있다.
- 이 `xml_files` 목록이 이후 데이터 분할과 Dataset 정의의 출발점이 된다.

## 5. XML 구조 확인

첫 번째 XML 파일을 들여쓰기해서 출력해 구조를 확인한다.

```python
import xml.dom.minidom

print(xml.dom.minidom.parse(xml_files[0]).toprettyxml())   # 첫 번째 XML을 들여쓰기해서 출력
```

```
<?xml version="1.0" ?>
<annotation>
    <folder>OXIIIT</folder>
    <filename>British_Shorthair_10.jpg</filename>
    <source>
        <database>OXFORD-IIIT Pet Dataset</database>
        <annotation>OXIIIT</annotation>
        <image>flickr</image>
    </source>
    <size>
        <width>233</width>
        <height>350</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>cat</name>
        <pose>Frontal</pose>
        <truncated>0</truncated>
        <occluded>0</occluded>
        <bndbox>
            <xmin>83</xmin>
            <ymin>29</ymin>
            <xmax>197</xmax>
            <ymax>142</ymax>
        </bndbox>
        <difficult>0</difficult>
    </object>
</annotation>
```

- 쓰는 태그는 세 가지다: `filename`(이미지 파일명), `object/name`(클래스), `object/bndbox`(박스 좌표).
- `size`에는 이미지의 너비 233, 높이 350, 채널 수 3이 들어 있다.
- 박스는 왼쪽 위 모서리(`xmin`, `ymin`)와 오른쪽 아래 모서리(`xmax`, `ymax`) 좌표 4개로 나타낸다.

## 6. XML 파싱 테스트

XML에서 클래스 이름과 박스 좌표를 꺼내는 코드를 첫 번째 파일로 먼저 시험한다.

```python
import xml.etree.ElementTree as ET

tree = ET.parse(xml_files[0])                              # XML 파일 읽기
root = tree.getroot()                                      # 최상위 태그(annotation)
label = root.find("object").find("name").text              # 클래스 이름
bndbox = root.find("object").find("bndbox")                # 박스 태그
bbox = [int(bndbox.find("xmin").text),                     # 문자열 좌표를 정수로 바꿔 담기
        int(bndbox.find("ymin").text),
        int(bndbox.find("xmax").text),
        int(bndbox.find("ymax").text)
]

print(bbox)
```

```
[83, 29, 197, 142]
```

- `find`로 태그를 찾고 `.text`로 안의 글자를 꺼낸다. XML의 값은 문자열이라 `int`로 바꿔야 숫자로 쓸 수 있다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
