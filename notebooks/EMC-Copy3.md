

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

### Данные взяты из открытого датасета: https://www.kaggle.com/joniarroba/noshowappointments
Набор представляет из себя записи 300000 пациентов, содержащих 15 признаков.

Необходимо создать модель, которая предсказывает вероятность того, придёт ли пациент на повторный визит или нет.


```python
dataset=pd.read_csv('No-show-Issue-Comma-300k.csv')
```

### Для обучения модели будут использованы 11 признаков:
1. Возраст пациента;
2. Пол пациента;
3. День недели, на окторую пациент записан;
4. Есть ли диабет;
5. Есть ли алкогольная зависимость;
6. Есть ли артериальная гипертензия;
7. Есть ли когнитивные нарушения;
8. Курение;
9. Образование;
10. Есть ли туберкулез;
12. Интервал между записью и временем приёма (дней);


```python
dataset.head()
```




<div>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>AppointmentRegistration</th>
      <th>ApointmentData</th>
      <th>DayOfTheWeek</th>
      <th>Status</th>
      <th>Diabetes</th>
      <th>Alcoolism</th>
      <th>HiperTension</th>
      <th>Handcap</th>
      <th>Smokes</th>
      <th>Scholarship</th>
      <th>Tuberculosis</th>
      <th>Sms_Reminder</th>
      <th>AwaitingTime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>M</td>
      <td>2014-12-16T14:46:25Z</td>
      <td>2015-01-14T00:00:00Z</td>
      <td>Wednesday</td>
      <td>Show-Up</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>24</td>
      <td>F</td>
      <td>2015-08-18T07:01:26Z</td>
      <td>2015-08-19T00:00:00Z</td>
      <td>Wednesday</td>
      <td>Show-Up</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>F</td>
      <td>2014-02-17T12:53:46Z</td>
      <td>2014-02-18T00:00:00Z</td>
      <td>Tuesday</td>
      <td>Show-Up</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>M</td>
      <td>2014-07-23T17:02:11Z</td>
      <td>2014-08-07T00:00:00Z</td>
      <td>Thursday</td>
      <td>Show-Up</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38</td>
      <td>M</td>
      <td>2015-10-21T15:20:09Z</td>
      <td>2015-10-27T00:00:00Z</td>
      <td>Tuesday</td>
      <td>Show-Up</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-6</td>
    </tr>
  </tbody>
</table>
</div>



### Предобработка данных:


```python
age_drop=dataset[dataset.Age<0]
dataset_1=dataset.drop(age_drop.index, axis=0)
```


```python
X=dataset_1.iloc[:, [0, 1,4, 6,7,8,9,10,11,12,13,14]].values
y=dataset_1.iloc[:, 5].values
```


```python
X[0]
```




    array([19, 'M', 0, 0, 0, 0, 0, 0, 0, 0, -29], dtype=object)




```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
binarizer_1=LabelEncoder()
X[:, 1]=binarizer_1.fit_transform(X[:, 1])
y=binarizer_1.fit_transform(y)

binarizer_2=LabelEncoder()
X[:, 2]=binarizer_4.fit_transform(X[:, 2])
one_hot_day_of_week=OneHotEncoder(categorical_features=[2])
X=one_hot_day_of_week.fit_transform(X).toarray()
X=X[:, 1:]
```


```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

    /home/nbuser/anaconda3_431/lib/python3.6/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)



```python
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
```

### Содание модели искусственной нейронной сети


```python
from keras.models import Sequential
from keras.layers import Dense, PReLU, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adagrad

rms1=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

classifier=Sequential()
classifier.add(Dense(units=32, kernel_initializer='uniform', input_dim=X_train.shape[1]))
classifier.add(PReLU())
classifier.add(Dense(units=16, kernel_initializer='uniform', input_dim=X_train.shape[1]))
classifier.add(PReLU())
classifier.add(Dense(units=8, kernel_initializer='uniform', input_dim=X_train.shape[1]))
classifier.add(PReLU())
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))


classifier.compile(optimizer=rms1, loss='binary_crossentropy', metrics=['accuracy'])
```

    Using TensorFlow backend.


### Обучение модели:


```python
classifier.fit(X_train, y_train, epochs=5, batch_size=1000, verbose=2, class_weight = 'auto')
```

    Epoch 1/5
    0s - loss: 0.5945 - acc: 0.7009
    Epoch 2/5
    1s - loss: 0.5947 - acc: 0.7011
    Epoch 3/5
    1s - loss: 0.5945 - acc: 0.7010
    Epoch 4/5
    1s - loss: 0.5947 - acc: 0.7011
    Epoch 5/5
    1s - loss: 0.5946 - acc: 0.7010





    <keras.callbacks.History at 0x7f5f3cda8cc0>



### Валидация модели на тестовой выборке (на данных, которые модель "не видела")


```python
y_pred=classifier.predict_classes(X_test, verbose=2)
```

### точность модели = количество правильных прогнозов / общее количество прогнозов


```python
accuracy=classifier.evaluate(X_test, y_test, batch_size=1, verbose=2)
print(' Точность модели: %.1f'% (accuracy[1]*100), "%")
```

     Точность модели: 69.6 %


### Точность модели составила: 69.6 %

### Для данного набора данных самая высокая точность модели на тестовой выборке составила 71% (из опубликованных результатов на сайте kaggle). Это можно объяснить: 
### 1. Искажением некоторых признаков: в переменной возраст встречались отрицательные и нулевые значения;
### 2. Нерелеватными признаками: теоретически, для обучения модели логичнее взять такие признаки, как: 
    - общее количество консультаций, 
    - количество бонусных баллов, 
    - общее количество отменённых визитов к врачу, 
    - длительность обслуживания пациента, 
    - количество госпитализаций и т.п. 
##### Релеватные признаки позволят повысить предиктивную точность модели.


```python

```
