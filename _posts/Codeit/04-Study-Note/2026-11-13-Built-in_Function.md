---
title: "Built-in Function - map, filter, zip, enumerate"
date: 2026-11-13 13:20:00 +0900
categories: [Codeit, '04-Study-Note']
tags: [codeit, 파이썬, python, 내장함수, map, filter, zip, enumerate, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---
### 1. `map(함수, 반복가능객체)` — 일괄 변환

- 모든 요소에 특정 함수를 똑같이 적용해 **변환**한다.
- 이터레이터(Iterator) 객체를 반환하므로 결과를 바로 확인하려면 `list()`로 감싸야 한다.
- `lambda`와 함께 쓰거나, `str`, `int`, `len` 같은 기존 내장 함수를 바로 넘길 때 유용하다.

📍[예시] 리스트의 모든 숫자를 **문자열(`str`)로 변환**하기

- 일반 `for`문

```python
nums = [1, 2, 3, 4]
str_nums = []

for x in nums:
    str_nums.append(str(x))
# 결과: ['1', '2', '3', '4']
```

- `map()` 함수

```python
nums = [1, 2, 3, 4]
str_nums = list(map(str, nums))
# 결과: ['1', '2', '3', '4']

# lambda와 함께 연산 적용
doubled = list(map(lambda x: x * 2, nums))
# 결과: [2, 4, 6, 8]
```

### 2. `filter(함수, 반복가능객체)` — 조건 필터링

- 함수의 반환값이 **`True`인 요소만 통과**시켜 걸러낸다.
- `map()`과 마찬가지로 결과를 리스트로 보려면 `list()`로 감싸야 한다.

📍[예시] **0보다 큰 양수만** 골라내기

- 일반 `for`문

```python
nums = [-2, -1, 0, 1, 2, 3]
positives = []

for x in nums:
    if x > 0:
        positives.append(x)
# 결과: [1, 2, 3]
```

- `filter()` 함수

```python
nums = [-2, -1, 0, 1, 2, 3]
positives = list(filter(lambda x: x > 0, nums))
# 결과: [1, 2, 3]
```

### 3. `zip(객체1, 객체2, ...)` — 여러 데이터 짝짓기

- 길이가 같은 여러 반복 가능 객체의 **같은 인덱스 위치끼리 튜플로 묶어준다.**
- 두 리스트를 합쳐서 `dict()`로 변환할 때 가장 많이 쓰인다.

📍[예시] 키(Key) 리스트와 값(Value) 리스트를 **딕셔너리로 합치기**

- 일반 `for`문

```python
keys = ["name", "age", "job"]
values = ["맥북", 5, "computer"]
profile = {}

for i in range(len(keys)):
    profile[keys[i]] = values[i]
# 결과: {'name': '맥북', 'age': 5, 'job': 'computer'}
```

- `zip()` 함수

```python
keys = ["name", "age", "job"]
values = ["맥북", 5, "computer"]

profile = dict(zip(keys, values))
# 결과: {'name': '맥북', 'age': 5, 'job': 'computer'}
```

### 4. `enumerate(반복가능객체, start=0)` — 번호(인덱스) 붙이기

- 데이터를 순회할 때 **인덱스 번호와 실제 값을 동시에** 꺼낸다.
- `start` 인자를 지정하면 시작 번호를 1번 등으로 바꿀 수 있다.

📍[예시] 과일 리스트에 **1번부터 번호를 매겨 딕셔너리**로 만들기

- 일반 `for`문

```python
fruits = ["사과", "바나나", "포도"]
fruit_dict = {}
idx = 1

for name in fruits:
    fruit_dict[idx] = name
    idx += 1
# 결과: {1: '사과', 2: '바나나', 3: '포도'}
```

- `enumerate()` 함수

```python
fruits = ["사과", "바나나", "포도"]
fruit_dict = dict(enumerate(fruits, start=1))
# 결과: {1: '사과', 2: '바나나', 3: '포도'}
```

### 5. `any(반복가능객체)` / `all(반복가능객체)` — 조건 만족 여부 확인

- `any()`: 요소 중 **하나라도 참(`True`)**이면 `True`를 반환한다.
- `all()`: 요소가 **모두 참(`True`)**이어야 `True`를 반환한다.

📍[예시] 시험 점수 리스트에서 **과락(60점 미만) 여부 및 전원 합격 여부** 확인하기

- 일반 `for`문

```python
scores = [85, 92, 58, 77]
has_fail = False

for s in scores:
    if s < 60:
        has_fail = True
        break
# 결과: True
```

- `any()` / `all()` 함수

```python
scores = [85, 92, 58, 77]

# 60점 미만이 하나라도 있는지 확인
has_fail = any(s < 60 for s in scores)   # 결과: True

# 모든 과목이 60점 이상인지 확인
all_pass = all(s >= 60 for s in scores)  # 결과: False
```

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 공부한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
