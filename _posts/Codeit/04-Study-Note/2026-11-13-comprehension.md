---
title: "Comprehension 4가지 종류 정리"
date: 2026-11-13 13:10:00 +0900
categories: [Codeit, '04-Study-Note']
tags: [codeit, 파이썬, python, 컴프리헨션, comprehension, 코드잇스프린트, AI, AI부트캠프, AI엔지니어, AI엔지니어취업]
math : true
---
### 1. 컴프리헨션(Comprehension)이란?

- `for`문과 `if`문을 한 줄로 축약해 새로운 자료구조를 생성하는 파이썬 고유 문법이다.
- 기본 구조: `[표현식 for 변수 in 반복가능객체 if 조건]`
- 빈 컨테이너를 만들고 `.append()`나 `.add()`를 반복하는 것보다 코드가 간결하고 실행 속도도 빠르다.
- 바깥을 감싸는 **괄호의 모양**과 **콜론(`:`) 유무**에 따라 4가지로 나뉜다.

🔍[비교] 4가지 컴프리헨션 요약

| 종류 | 기호 | 반환 타입 | 특징 |
|:---:|:---:|:---:|:---:|
| **리스트 컴프리헨션** | `[x for x in ...]` | `list` | 가장 많이 사용, 순서 유지 |
| **딕셔너리 컴프리헨션** | `{k: v for k, v in ...}` | `dict` | 콜론(`:`)으로 키-값 쌍 지정 |
| **세트 컴프리헨션** | `{x for x in ...}` | `set` | 중복 자동 제거, 순서 없음 |
| **제너레이터 표현식** | `(x for x in ...)` | `generator` | 튜플 아님, 필요할 때마다 하나씩 생성 |

### 2. 리스트 컴프리헨션 (`[ ... ]`)

- 대괄호 `[ ]` 안에 연산식과 반복문을 적어 새로운 **리스트(`list`)**를 만든다.

📍[예시] 1부터 5까지 숫자 중 **짝수만 골라 제곱**하기

- 일반 `for`문
```python
nums = [1, 2, 3, 4, 5]
result = []

for x in nums:
    if x % 2 == 0:
        result.append(x ** 2)
# 결과: [4, 16]
```

- 리스트 컴프리헨션
```python
nums = [1, 2, 3, 4, 5]
result = [x ** 2 for x in nums if x % 2 == 0]
# 결과: [4, 16]
```

### 3. 딕셔너리 컴프리헨션 (`{ key: value ... }`)

- 중괄호 `{ }` 안에 **`키: 값`** 형태로 콜론(`:`)을 넣어 새로운 **딕셔너리(`dict`)**를 만든다.

📍[예시] 점수가 **80점 이상인 과목만 딕셔너리**로 추리기

- 일반 `for`문
```python
scores = {"국어": 90, "영어": 75, "수학": 85}
passed = {}

for k, v in scores.items():
    if v >= 80:
        passed[k] = v
# 결과: {'국어': 90, '수학': 85}
```

- 딕셔너리 컴프리헨션
```python
scores = {"국어": 90, "영어": 75, "수학": 85}
passed = {k: v for k, v in scores.items() if v >= 80}
# 결과: {'국어': 90, '수학': 85}
```

### 4. 세트(집합) 컴프리헨션 (`{ ... }`)

- 중괄호 `{ }` 안에 콜론(`:`) 없이 단일 값만 적어 **중복이 제거된 집합(`set`)**을 만든다.

📍[예시] 단어 목록에서 **첫 글자만 중복 없이 추출**하기

- 일반 `for`문
```python
words = ["apple", "banana", "avocado", "cherry"]
first_letters = set()

for w in words:
    first_letters.add(w[0])
# 결과: {'a', 'b', 'c'}
```

- 세트 컴프리헨션
```python
words = ["apple", "banana", "avocado", "cherry"]
first_letters = {w[0] for w in words}
# 결과: {'a', 'b', 'c'}
```

### 5. 제너레이터 표현식 (`( ... )`)

- 소괄호 `( )`를 사용하며, **튜플 컴프리헨션이 아님에 주의**해야 한다.
- 결과를 메모리에 한꺼번에 올리지 않고, 요청할 때마다 **값을 하나씩 생성**한다.
- 대용량 데이터의 합계(`sum`)나 최댓값(`max`)을 구할 때 메모리를 크게 절약할 수 있다.

📍[예시] 1부터 100만까지 **제곱의 합** 구하기

- 일반 `for`문
```python
total = 0

for x in range(1, 1000001):
    total += x ** 2
```

- 제너레이터 표현식
```python
# 리스트 [...]를 만들지 않고 하나씩 꺼내 바로 더함
total = sum(x ** 2 for x in range(1, 1000001))

# next()로 하나씩 꺼내 쓰기
gen = (x * 2 for x in [10, 20, 30])
print(next(gen))  # 출력: 20
print(next(gen))  # 출력: 40
```

&nbsp;

&nbsp;

\- 이 글은 Codeit 강의를 수강하며 학습한 내용을 바탕으로 개인적으로 공부한 것입니다. ([Codeit 공식 사이트](https://codeit.kr/)) -
{: .small .text-muted .text-center }
