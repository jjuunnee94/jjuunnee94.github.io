---
title: "Object Detection SSD 02"
date: 2026-11-25 10:00:00 +0900
categories: [Codeit, 04-Study-Note]
tags: [codeit, mission7, object-detection, ssd, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math: true
---

## 1. 이미지 열기

XML 안의 `filename`으로 이미지를 찾아 연다.

```python
img_name = root.find("filename").text                      # XML에 적힌 이미지 파일명
print(img_name)

img_path = os.path.join(image_dir, img_name)               # 이미지 경로

from PIL import Image
img = Image.open(img_path)                                 # 이미지 열기
```

```
British_Shorthair_10.jpg
```

## 2. 이미지와 바운딩 박스 시각화

이미지 위에 정답 박스를 그려, 읽어 온 좌표가 맞는지 눈으로 확인한다.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()
ax.imshow(img)                                             # 이미지 그리기
xmin, ymin, xmax, ymax = bbox                              # 박스 좌표 풀기
width = xmax - xmin                                        # 박스 너비
height = ymax - ymin                                       # 박스 높이
rect = patches.Rectangle((xmin, ymin), width, height, lw=2, edgecolor="red", facecolor="none")  # 시작점, 너비, 높이
ax.add_patch(rect)                                         # 이미지 위에 박스 추가
plt.show()
```

`Rectangle`은 시작점(왼쪽 위)과 너비·높이를 받으므로 좌표 4개를 아래 식으로 바꿔서 넘긴다.

| 이름 | 말로 푼 식 | 기호 식 |
|:---:|:---:|:---:|
| 너비 | 오른쪽 x좌표 − 왼쪽 x좌표 | $w = x_{max} - x_{min}$ |
| 높이 | 아래쪽 y좌표 − 위쪽 y좌표 | $h = y_{max} - y_{min}$ |

## 3. 클래스 확인

모든 XML을 훑어 클래스 이름을 모은다. `set`을 쓰면 중복이 자동으로 빠진다.

```python
classes = set()                                            # 중복 없는 클래스 모음
for xml_file in xml_files:
    root = ET.parse(xml_file).getroot()                    # XML 읽기
    for obj in root.findall("object"):                     # 객체마다 반복
        name = obj.find("name").text
        classes.add(name)                                  # 클래스 이름 추가

print("클래스 목록:", sorted(classes))
```

```
클래스 목록: ['cat', 'dog']
```

## 4. 클래스 이름 ↔ 번호 변환표

모델은 글자가 아닌 숫자를 쓰므로 이름과 번호를 서로 바꾸는 딕셔너리를 만든다.

```python
label2idx = {name: idx for idx, name in enumerate(sorted(classes), start=1)}   # 이름 → 번호
idx2label = {idx: name for idx, name in enumerate(sorted(classes), start=1)}   # 번호 → 이름

print(label2idx)
print(idx2label)
```

```
{'cat': 1, 'dog': 2}
{1: 'cat', 2: 'dog'}
```

- `start=1`로 번호를 1부터 매긴다. 0번은 배경(background)이 쓴다.

## 5. 데이터 무결성 점검

이미지가 열리는지, 박스 좌표가 정상인지 전체 파일을 검사한다.

```python
bad_files = []                                             # 문제 있는 파일 목록

for xml_path in xml_files:
    try:
        root = ET.parse(xml_path).getroot()                # XML 읽기
        img_name = root.find("filename").text              # 이미지 파일명
        img_path = os.path.join(image_dir, img_name)       # 이미지 경로

        with Image.open(img_path) as img:
            img.verify()                                   # 손상된 이미지인지 확인

        with Image.open(img_path) as img:                  # verify 뒤에는 다시 열어야 함
            w, h = img.size                                # 이미지 너비, 높이

        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax > w or ymax > h:   # 좌표 이상치
                bad_files.append((xml_path, "좌표 이상치"))

    except Exception as e:                                 # 파일·파싱 오류
        bad_files.append((xml_path, f"파일/파싱 오류: {e}"))

print("점검 완료! 불량 파일 개수:", len(bad_files))
print("불량 파일 목록:", bad_files[:10])
```

```
점검 완료! 불량 파일 개수: 0
불량 파일 목록: []
```

- 좌표 이상치 조건: 왼쪽이 오른쪽보다 크거나 같음, 위쪽이 아래쪽보다 크거나 같음, 음수 좌표, 이미지 크기 밖으로 나감.

## 6. 학습/검증 데이터 분할

XML 목록을 학습용 80%, 검증용 20%로 나눈다.

```python
from sklearn.model_selection import train_test_split

train_xmls, val_xmls = train_test_split(xml_files, test_size=0.2, random_state=42)   # 학습 80% / 검증 20%
```

- `random_state=42`로 섞는 순서를 고정해 다시 실행해도 같은 분할이 나온다.

&nbsp;

&nbsp;

이 글은 Codeit 강의를 수강하며 수행한 미션을 바탕으로 개인적으로 정리한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/))
{: .small .text-muted .text-center }
