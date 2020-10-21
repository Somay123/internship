# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:48:45 2019

@author: somay garg
"""
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

#Converting the dataframe into array
training = np.array(data_train,dtype='float')
testing = np.array(data_test,dtype='float')

#Single Data Visualization
plt.imshow(training[3,1:].reshape(28,28))

#More Visualization
W_Grid = 17
L_Grid = 21
fig , axes = plt.subplot(L_Grid,W_Grid,figsize=(17,17))
axes = axes.ravel()
n_training = len(training)
for i in np.arange(0,W_Grid*L_Grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(training[index,1:].reshape(28,28))
    
#Data Divison
x_train = training[:,1:]
y_train = training[:,0]

x_test = testing[:,1:]
y_test = testing[:,0]


x_train = x_train.reshape(x_train.shape[0],*(28,28,1))
x_test = x_test.reshape(x_test.shape[0],*(28,28,1))


#Training the Model
import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

model = Sequential()
model.add(Conv2D(32,3,3, input_shape=(28,28,1) , activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
model.fit(x_train,y_train,batch_size=512,epochs=20)

y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix
avg = confusion_matrix(y_test,y_pred)








