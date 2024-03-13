---
layout: single
title: "혼공머신 Ch.4-1 로지스틱 회귀"
---


# chapter 4-1 로지스틱 회귀 

이번에 만들 상품은 럭키백으로 각가의 확률을 제공하는 상품을 만들려고 한다.

이전에 배운 적이 있는 k-최근접이웃으로 각각의 확률을 제공하는 모델을 먼저 만들어보자.

# 1. k-최근접 이웃

## Step1. 데이터 준비하기




```python
# Load data
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Species</th>
      <th>Weight</th>
      <th>Length</th>
      <th>Diagonal</th>
      <th>Height</th>
      <th>Width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bream</td>
      <td>242.0</td>
      <td>25.4</td>
      <td>30.0</td>
      <td>11.5200</td>
      <td>4.0200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bream</td>
      <td>290.0</td>
      <td>26.3</td>
      <td>31.2</td>
      <td>12.4800</td>
      <td>4.3056</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bream</td>
      <td>340.0</td>
      <td>26.5</td>
      <td>31.1</td>
      <td>12.3778</td>
      <td>4.6961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bream</td>
      <td>363.0</td>
      <td>29.0</td>
      <td>33.5</td>
      <td>12.7300</td>
      <td>4.4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bream</td>
      <td>430.0</td>
      <td>29.0</td>
      <td>34.0</td>
      <td>12.4440</td>
      <td>5.1340</td>
    </tr>
  </tbody>
</table>
</div>



어떤 종류의 생선이 있는지 살펴보자.


```python
print(pd.unique(fish['Species']))
```

    ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']
    

데이터프레임에서 Species 열을 target으로 하고 Weight부터 Width까지 열을 feature로 사용하자.

데이터프레임에서 열을 선택하는 방법은 원하는 열을 리스트로 나열하는 것이다.

단, 열을 하나만 선택할 때는 리스트가 아닌 문자열 하나만 입력한다.  *리스트로 전달하면 2차원 배열이 된다.


```python
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])
```

    [[242.      25.4     30.      11.52     4.02  ]
     [290.      26.3     31.2     12.48     4.3056]
     [340.      26.5     31.1     12.3778   4.6961]
     [363.      29.      33.5     12.73     4.4555]
     [430.      29.      34.      12.444    5.134 ]]
    


```python
fish_target = fish['Species'].to_numpy()
```

### 데이터 분리하기



```python
# Split data to train and test
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
```

### 스케일링


```python
# scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

## Step2. 모델 훈련 및 평가


```python
# train model 
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

# evaluation
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```

    0.8907563025210085
    0.85
    


```python
# order of sample
print(kn.classes_)
```

    ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
    


```python
print(kn.predict(test_scaled[:5]))
```

    ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']
    


```python
# Check Probability value
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
```

    [[0.     0.     1.     0.     0.     0.     0.    ]
     [0.     0.     0.     0.     0.     1.     0.    ]
     [0.     0.     0.     1.     0.     0.     0.    ]
     [0.     0.     0.6667 0.     0.3333 0.     0.    ]
     [0.     0.     0.6667 0.     0.3333 0.     0.    ]]
    


```python
distances, indexes = kn.kneighbors(test_scaled[3:4])
```


```python
print(train_target[indexes])
```

    [['Roach' 'Perch' 'Perch']]
    

잘 예측을 해냈음을 볼 수 있다. 그러나 k-최근접 이웃은 확률이 0, 1/3, 2/3, 3/3 뿐이다. 조금 더 확률적인 의미를 가질 수 있도록 로지스틱 회귀를 이용해보자.

# 2. 로지스틱 회귀

## 2-1. 소개

**로지스틱 회귀(logistic regression)**은 이름은 회귀이지만 분류 모델이다. 이 알고리즘은 선형 회귀와 동일하게 선형 방정식을 학습한다. 우리가 가진 데이터로 설명하면 다음의 선형 방정식을 학습한다.

$$ z = a \times (Weight) + b \times (Length) + c \times (Diagonal) + d \times (Height) + e \times (Width) + f \\where, \; a, b, c, d, e = coefficient(weight)$$

z는 어떤 값이던 나올 수 있다. 하지만, 확률이 되려면 0~1 사이 값이 되어야 한다. 여기서 **시그모이드 함수(sigmoid function)**(또는 **로지스틱 함수(logistic function)**)을 사용한다. 

## 2-2. 시그모이드 함수

![%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C.jpg](%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C.jpg)



## 2-3. 로지스틱 회귀로 이진 분류 수행하기

### Step1. 데이터 준비하기

불리언 인덱싱을 이용해서 도미와 빙어의 행만을 골라내자.


```python
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]
```

###  Step2. 모델 훈련하기 


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
```




    LogisticRegression()




```python
# check some of predict 
print(lr.predict(train_bream_smelt[:5]))
```

    ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
    


```python
# check Probability value
print(lr.predict_proba(train_bream_smelt[:5]))
```

    [[0.99759855 0.00240145]
     [0.02735183 0.97264817]
     [0.99486072 0.00513928]
     [0.98584202 0.01415798]
     [0.99767269 0.00232731]]
    

첫 행은 음성 클래스(0)에 대한 확률이고 두 번째 열은 양성 클래스(1)에 대한 확률이다. 

Bream과 Smelt 중 어떤 것이 양성인지 알아보자.

k-최근접 이웃 분류기에서 본 것처럼 사이킷런은 타깃값을 알파벳순으로 정렬해서 사용한다.
classes_ 속성을 이용하면 다음을 확인할 수 있다.


```python
print(lr.classes_)
```

    ['Bream' 'Smelt']
    

* 만약 Bream을 양성 클래스로 이용하려면 Bream 타깃값을 1로 만들고 나머지 타깃값은 0으로 만들어 사용하면 된다.

로지스틱 회귀가 학습한 계수를 확인해보자.


```python
print(lr.coef_, lr.intercept_)
```

    [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
    

위의 정보를 통해 로지스틱 회귀 모델이 학습한 방정식은 다음의 식이다.

\begin{align*}
z = -0.4037798 \times (Weight)  -0.57620209 \times (Length) -0.66280298 \times (Diagonal) -1.01290277 \times (Height) -0.73168947\times (Width)  -2.16155132
\end{align*}

LogisticRegression 모델로 z값을 계산한 결과는 decision_function() 메서드로 출력할 수 있다. 처음 5개의 샘플의 z값을 출력해보자.


```python
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
```

    [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]
    

여기서 얻은 z값들을 시그모이드 함수에 통과시키면 확률을 얻을 수 있다. scipy라이브러리에 시그모이드 함수를 이용해서 확률로 변환해보자.

* np.exp() 함수를 이용할 수 있으나 라이브러리 이용이 훨씬 편리하고 안전하다.


```python
from scipy.special import expit
print(expit(decisions))
```

    [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]
    

위에서 얻은 확률 값은 앞서 predict_proba()메서드로 확인한 양성 클래스의 확률값과 일치함을 알 수 있다. 즉, decision_function()메서드는 양성 클래스의 z값을 반환한다.

summary

로지스틱 회귀를 이용한 이진 분류
preict_proba() : 음성 클래스 | 양성 클래스의 확률 출력
decision_function() : 양성 클래스에 대한 z값 출력
coef_ 속성, intercept_ 속성 : 로지스틱 모델이 학습한 선형 방정식의 계수 

## 2-4. 로지스틱 회귀로 다중 분류 수행하기

이제 LogisticRegression 클래스를 이용해서 7가지 종류의 생선을 분류해보자.

### 소개

LogisticRegression 클래스는 기본적으로 반복적인 알고리즘을 사용한다. 반복 횟수를 지정하는 매개변수로 max_iter가 있고 기본값은 100이다. 

또 LogisticRegression은 기본저긍로 릿지 회귀와 같이 계수의 제곱을 규제한다. 릿지 회귀에서는 alpha 값을 조절해서 규제를 조절하지만, LogisticRegression은 C 매개변수를 통해 규제를 조절한다. 기본값은 1이고 릿지 회귀와 반대로 값이 커지면 규제의 정도가 약해지고 값이 작아지면 규제의 정도가 강해진다.

데이터는 이미 준비되어 있으므로 바로 모델에 적용해서 훈련시키자.

### Step1. 모델 훈련


```python
# train model
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
```




    LogisticRegression(C=20, max_iter=1000)




```python
# evaluation
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

    0.9327731092436975
    0.925
    


```python
# check prediction sample
print(lr.predict(test_scaled[:5]))
```

    ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
    


```python
# check probability values of prediction sample
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
```

    [[0.    0.014 0.841 0.    0.136 0.007 0.003]
     [0.    0.003 0.044 0.    0.007 0.946 0.   ]
     [0.    0.    0.034 0.935 0.015 0.016 0.   ]
     [0.011 0.034 0.306 0.007 0.567 0.    0.076]
     [0.    0.    0.904 0.002 0.089 0.002 0.001]]
    

이진 분류와의 차이점은 각 샘플에 대해서 7개의 예측값이 나왔다는 것이다.

각각의 열이 어떤 물고기를 나타내는지 classes_속성에서 클래스 정보를 확인해보자.


```python
print(lr.classes_)
```

    ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
    

다중 분류는 클래스 개수만큼 확률을 출력한다. 이 중 가장 높은 확률이 예측 클래스이다.

그럼 선형 방정식은 어떤 모습인지 확인해보자.


```python
print(lr.coef_.shape, lr.intercept_.shape)
```

    (7, 5) (7,)
    

5개의 특성을 이용하므로 coef_ 배열의 열은 5개이다. 그런데, coef_ 배열에 행과 intercept_의 행이 7개가 있다는 것은 z를 7개나 계산한다는 의미이다.이는 곧 다중 분류는 클래스마다 z값을 하나씩 계산하고 z값을 출력하는 클래스가 예측 클래스가 된다는 의미이다.

즉, 이 경우에 한가지 샘플이 들어오면 7가지 종류의 물고기일 확률을 계산하고 그 중 가장 큰 확률인 것이 예측하는 물고기 종류이다.

그럼 다중 분류에서는 확률을 어떻게 계산할까? 다중 분류에서는 **소프트맥스(softmax)** 함수를 사용하여 z값을 확률로 변환하다.

#### 소프트맥스 함수

소프트맥스 함수는 다음과 같은 방식으로 z값을 확률로 변환한다.

$$e\_sum = e^{z1} + e^{z2} + e^{z3} + ... + e^{zn}$$
$$s1 = \frac{e^{z1}}{e\_sum}, \quad s2 = \frac{e^{z2}}{e\_sum}, \quad ... sn = \frac{e^{zn}}{e\_sum}$$

이제 이 과정을 코딩으로 직접 계산해보자.


```python
# z value
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
```

    [[ -6.5    1.03   5.16  -2.73   3.34   0.33  -0.63]
     [-10.86   1.93   4.77  -2.4    2.98   7.84  -4.26]
     [ -4.34  -6.23   3.17   6.49   2.36   2.42  -3.87]
     [ -0.68   0.45   2.65  -1.19   3.26  -5.75   1.26]
     [ -6.4   -1.99   5.82  -0.11   3.5   -0.11  -0.71]]
    


```python
# softmax functino
from scipy.special import softmax
proba = softmax(decision, axis=1) # axis=1 없으면 배열 전체에 대한 소프트맥스를 계산함.
print(np.round(proba, decimals=3))
```

    [[0.    0.014 0.841 0.    0.136 0.007 0.003]
     [0.    0.003 0.044 0.    0.007 0.946 0.   ]
     [0.    0.    0.034 0.935 0.015 0.016 0.   ]
     [0.011 0.034 0.306 0.007 0.567 0.    0.076]
     [0.    0.    0.904 0.002 0.089 0.002 0.001]]
    


```python
a = softmax(decision) # axis=1 없으면 배열 전체에 대한 소프트맥스를 계산함.
print(np.round(a, decimals=3))
```

    [[0.    0.001 0.044 0.    0.007 0.    0.   ]
     [0.    0.002 0.029 0.    0.005 0.633 0.   ]
     [0.    0.    0.006 0.164 0.003 0.003 0.   ]
     [0.    0.    0.004 0.    0.007 0.    0.001]
     [0.    0.    0.084 0.    0.008 0.    0.   ]]
    


```python

```
