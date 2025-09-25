import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


#특정 데이터 셋을 train, test set으로 나눠주는 함수
def Divide_data_set(new_xy_np,train,test): 
  
    
    #가져온 데이터의 비율을 구함
    rate=len(new_xy_np)//10
    
    #데이터 나눌 구간
    train_rate=train*rate
    test_rate=test*rate+train_rate
    
    #인덱스 셔플
    np.random.shuffle(new_xy_np)
    
    #데이터set 나누기
    Training_set=new_xy_np[0:train_rate,:]
    Test_set=new_xy_np[train_rate:test_rate,:]
    data=pd.DataFrame(Training_set)

    
    Training_set=data.to_numpy(dtype="float32")#데이터 프레임을 넘파이 배열로 형변환
    data=pd.DataFrame(Test_set)


    Test_set=data.to_numpy(dtype="float32")#데이터 프레임을 넘파이 배열로 형변환
    
    
    return Training_set,Test_set

#one-hot encoding함수
def One_Hot_Encoding(data_y):
    N=len(data_y)#데이터의 총 개수
    Q_class=np.unique(data_y)#클래스 종류
    Q=len(Q_class)#클래스 개수
    
    #one-hot-y초기화
    one_hot_y=np.zeros([N,Q])
    #각 클래스에 대해 One-Hot 표현으로 변환
    for k in np.arange(0,N,1):
        #해당 비교값이 자동으로 float형으로 변환 되어 저장됨(넘파이 배열 특징)
        one_hot_y[k,:]=(data_y[k]==Q_class)
        
    return one_hot_y
data=pd.read_csv("C:\\Users\\USER\\Downloads\\NN_data.csv")
data=np.array(data, dtype=np.float32)
train=7
test=3
train,test=Divide_data_set(data.copy(), train, test)
x_train=train[:,:3]
t_train=train[:,3]
x_test=test[:,:3]
t_test=test[:,3]
t_train=One_Hot_Encoding(t_train)
t_test=One_Hot_Encoding(t_test)
model=keras.models.Sequential([
    keras.layers.Dense(units= 1024, activation='relu'),
    keras.layers.Dense(units= 1024, activation='relu'),
    keras.layers.Dense(units= 1024, activation='relu'),
    keras.layers.Dense(units= 512, activation='relu'),
    keras.layers.Dense(units= 6, activation='softmax'),
    ])

optimizer=keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=["accuracy"])

history = model.fit(x_train,t_train,epochs=100,validation_data = (x_test, t_test))

pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim(0.1)
plt.show()
