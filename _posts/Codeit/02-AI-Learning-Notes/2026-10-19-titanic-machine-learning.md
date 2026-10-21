---
title: "Predicting Titanic Survival Rates using Machine Learning"
date: 2026-10-19 19:00:00 +0900
categories: [Codeit, '02-AI-Learning-Notes']
tags: [codeit, journal]
---

### 타이타닉 생존자 예측

머신러닝 모델을 구축하기에 앞서 가장 먼저 해야 할 일은 **데이터의 구조를 탐색(EDA)하고 올바른 전처리 방향을 설정하는 것**입니다.

본 실습에서는 제공된 데이터를 바탕으로 다음과 같이 데이터셋을 정의하고 탐색합니다.
- **학습 데이터 (Train Data)** : 모델을 학습시키는 데 사용되는 데이터
- **검증 데이터 (Validation Data)** : 학습 중 과적합(Overfitting)을 방지하고 성능을 평가하기 위해 학습 데이터 내부에서 분할한 데이터
- **테스트 데이터 (Test Data)** : 최종 모델의 예측 성능을 측정하기 위한 평가 데이터

전처리 과정에서는 예측할 핵심 목표인 <b>타깃(Target: Survived)</b>과 이에 영향을 미치는 다양한 <b>특성(Feature)</b> 간의 관계를 파악하고, 모델이 학습하기 최적의 형태로 데이터를 가공합니다.

### 1. 데이터 확인 및 탐색 (EDA)

데이터의 구조와 기본 정보를 확인하기 위해 사용되는 주요 메서드는 다음과 같습니다.

- `display()` : 괄호 안의 객체를 시각적으로 출력
- `.shape` : 데이터의 행(데이터 개수)과 열(특성 개수) 크기를 `(행, 열)` 형태로 반환
- `.head()` : 상위 N개의 데이터를 출력 (기본값: 5개)
- `.info()` : 각 특성의 이름, 결측치를 제외한 데이터 개수, 데이터 타입을 출력

```python
# 학습 데이터 구조 확인
display(train_df.shape)
display(train_df.head())
display(train_df.info())

# 테스트 데이터 구조 확인
display(test_df.shape)
display(test_df.head())
display(test_df.info())
```
### 2. 결측치 처리(Inputation)
```python
age_median = train_df['Age'].median()        # 나이 중간값
fare_median = train_df['Fare'].median()      # 요금 중간값
most_embarked = train_df['Embarked'].mode()[0]  # 탑승 항구 최빈값

def fill_missing_values(df):
    df = df.copy()
    df['Age'] = df['Age'].fillna(age_median)
    df['Fare'] = df['Fare'].fillna(fare_median)
    df['Embarked'] = df['Embarked'].fillna(most_embarked)
    return df

# 전처리 적용
train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)
```
### 3. 파생 변수 생성(Feature Engineerting)
```python
def new_data(df):
  df = df.copy() # 데이터가 덮어씌어지지않게 복사
  df['Family'] = 1 + df['SibSp'] + df['Parch'] # 배우자 및 가족으로 가족 총 인원수 계산(1은 본인)
  return df

train_df = new_data(train_df)
test_df = new_data(test_df)
```
### 4. 연령대 구간 및 라벨 정의(Encoding)
```python
bins = [0, 10, 20, 30, 50, 100]
labels = [0, 1, 2, 3, 4]

def transform_features(df):
    df = df.copy()
    df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True).astype(int)
    df['Cabin'] = np.where(df['Cabin'].isna(), 0, 1)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

# 전처리 적용
train_df = transform_features(train_df)
test_df = transform_features(test_df)
```
### 5. 학습에 사용할 특성 선택
```python
features = ['Pclass', 'Sex', 'Age', 'Family', 'Cabin', 'Embarked', 'AgeGroup', 'Fare']
X = train_df[features]
y = train_df['Survived']

# 학습 데이터와 검증 데이터 분할 (8:2 비율)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)
```
### 6. 모델 실험
```python
# 기본 결정 트리(DecisionTree)
model = DecisionTreeClassifier(random_state=41) # 기본 결정 트리 고정시드 41
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy') # 결정 트리 모델 5-Fold 교차 검증 수행

print("기본 결정 트리 교차 검증 정확도 목록:", scores)
print("기본 결정 트리 평균 정확도:", scores.mean())


# 랜덤 포레스트(RandomForest)
model_rf = RandomForestClassifier(random_state=41) # 트리 개수100개 고정시드 41인 RF모델
scores_rf = cross_val_score(model_rf, X, y, cv=5, scoring='accuracy') # RF 모델 5-Fold 교차 검증 수행

print("랜덤 포레스트 교차 검증 정확도 목록:", scores_rf)
print("랜덤 포레스트 평균 정확도:", scores_rf.mean())


# 그래디언트 부스팅(GradientBoosting)
model_gb = GradientBoostingClassifier(random_state=41)
scores_gb = cross_val_score(model_gb, X, y, cv=5, scoring='accuracy')
print("그래디언트 부스팅 검증 정확도 목록:", scores_gb)
print("그래디언트 부스팅 평균 정확도:", scores_gb.mean())
```
# 출력값  
기본 결정 트리 교차 검증 정확도 목록: [0.73743017 0.75842697 0.84269663 0.76966292 0.82022472]  
기본 결정 트리 평균 정확도: 0.7856882807105643  

랜덤 포레스트 교차 검증 정확도 목록: [0.77653631 0.80898876 0.84269663 0.78089888 0.85955056]  
랜덤 포레스트 평균 정확도: 0.8137342288619672  

그래디언트 부스팅 검증 정확도 목록: [0.81564246 0.82022472 0.85955056 0.80898876 0.85393258]  
그래디언트 부스팅 평균 정확도: 0.8316678174628084 

### 7. 하이퍼파라미터 튜닝

```python
# 교차검증설정
from sklearn.model_selection import StratifiedKFold
cv_splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=41) # 5묶음으로 나누고 무작위로 섞은뒤 고정시드 41
```

```python
# 기준모델로 시작
gb_rate1d3 = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=41)
scores_gb_rate1d3 = cross_val_score(gb_rate1d3, X, y, cv=cv_splits, scoring='accuracy')

print("lr=0.1, max_depth=3 GB CV 평균 정확도:", scores_gb_rate1d3.mean())
```
lr=0.1, max_depth=3 GB CV 평균 정확도: 0.8271922666499278  
lr 0.05 감소

```python
gb_rate05d3 = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=41)
scores_gb_rate05d3 = cross_val_score(gb_rate05d3, X, y, cv=cv_splits, scoring='accuracy')

print("lr=0.05, max_depth=3 GB CV 평균 정확도:", scores_gb_rate05d3.mean())
```
lr=0.05, max_depth=3 GB CV 평균 정확도: 0.8249450756386919  
lr 0.1 복원, max_depth 5 증가

```python
gb_rate1d5 = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=41)
scores_gb_rate1d5 = cross_val_score(gb_rate1d5, X, y, cv=cv_splits, scoring='accuracy')

print("lr=0.1, max_depth=5 GB CV 평균 정확도:", scores_gb_rate1d5.mean())
```
lr=0.1, max_depth=5 GB CV 평균 정확도: 0.8249325214989642  
max_depth 5 유지, lr 0.05 감소

```python
gb_rate05d5 = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=41)
scores_gb_rate05d5 = cross_val_score(gb_rate05d5, X, y, cv=cv_splits, scoring='accuracy')

print("lr=0.05, max_depth=5 GB CV 평균 정확도:", scores_gb_rate05d5.mean())
```
lr=0.05, max_depth=5 GB CV 평균 정확도: 0.8339338396836358  
랜덤 포레스트로 변경 (max_depth=5, n_estimators=100)

```python
model_n100d5 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=41)
scores_n100d5 = cross_val_score(model_n100d5, X, y, cv=cv_splits, scoring='accuracy')

print("max_depth=5, n_estimators=100 적용 평균 정확도:", scores_n100d5.mean())
```
max_depth=5, n_estimators=100 적용 평균 정확도: 0.8260812252840374  
n_estimators 200 증가

```python
model_n200d5 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=41)
scores_n200d5 = cross_val_score(model_n200d5, X, y, cv=cv_splits, scoring='accuracy')

print("max_depth=5, n_estimators=200 적용 평균 정확도:", scores_n200d5.mean())
```
max_depth=5, n_estimators=200 적용 평균 정확도: 0.8249450756386919  
n_estimators 100 복원, max_depth 7 증가
#
```python
model_n100d7 = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=41)
scores_n100d7 = cross_val_score(model_n100d7, X, y, cv=cv_splits, scoring='accuracy')

print("max_depth=7, n_estimators=100 적용 평균 정확도:", scores_n100d7.mean())
```
max_depth=7, n_estimators=100 적용 평균 정확도: 0.8182097796748478  
max_depth 3 감소

```python
model_n100d3 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=41)
scores_n100d3 = cross_val_score(model_n100d3, X, y, cv=cv_splits, scoring='accuracy')

print("max_depth=3, n_estimators=100 적용 평균 정확도:", scores_n100d3.mean())
```
max_depth=3, n_estimators=100 적용 평균 정확도: 0.8024982738057875

### 8. 최종 평가
```python
# 모델별 정확도 그래프화
names = [
    'GB (lr=0.1, depth=3)',
    'GB (lr=0.05, depth=3)',
    'GB (lr=0.1, depth=5)',
    'GB (lr=0.05, depth=5)',
    'RF (depth=5, n=100)',
    'RF (depth=5, n=200)',
    'RF (depth=7, n=100)',
    'RF (depth=3, n=100)'
]

scores_list = [
    scores_gb_rate1d3.mean(),
    scores_gb_rate05d3.mean(),
    scores_gb_rate1d5.mean(),
    scores_gb_rate05d5.mean(),
    scores_n100d5.mean(),
    scores_n200d5.mean(),
    scores_n100d7.mean(),
    scores_n100d3.mean()
]

plt.figure(figsize=(10, 5))
plt.barh(names, scores_list)
plt.xlim(0.80, 0.84)
plt.show()
```
![타이타닉 정확도 그래프](/assets/titanic-accuracy-graph.png)

```python
# 최종 모델 설정
best_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=41)
best_scores = cross_val_score(best_model, X, y, cv=cv_splits, scoring='accuracy')

# 최종 모델 출력
print("학습검증데이터 기반\n")
print("최종모델 : 학습률 0.05 트리 깊이 5의 그래디언트 부스팅 모델")
print("최종모델 검증 정확도 목록:", best_scores)
print("최종모델 적용 평균 정확도:", best_scores.mean(),"\n")
```
학습검증데이터 기반

최종모델 : 학습률 0.05 트리 깊이 5의 그래디언트 부스팅 모델
최종모델 검증 정확도 목록: [0.81564246 0.82022472 0.85955056 0.80898876 0.85393258]
최종모델 적용 평균 정확도: 0.8339338396836358 

테스트데이터 기반

![최종 테스트 결과](/assets/final-test-result.png)

최종모델 : 트리갯수 50 트리 깊이 5의 랜덤포레스트 모델  
그래디언트 부스팅은 과적합되기 쉽기 때문에 학습검증정확도는 높게 나왔으나 테스트정확도는 낮게 나왔다.  
트리 깊이 또한 깊을수록 노이즈까지 과도하게 학습할 수 있으므로 트리깊이를 낮추니 학습검증정확도는 낮아졌으나 테스트정확도는 높아졌다.  
랜덤포레스트는 그래디언트 부스팅에 비해 학습검증정확도는 낮았으나 노이즈에 안정적이므로 테스트데이터는 가장 높게 나왔다.  
