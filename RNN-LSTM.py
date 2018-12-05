# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 14:58:40 2018

@author: Anoop
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

product_id_ds = pd.read_csv('key_product_IDs.csv')
product_ids = product_id_ds.iloc[:, 0:1].values


def predictFunc(indx,prd_num):
    dataset_train = pd.read_csv('Training_Set_Full.csv')
    dataset_test = pd.read_csv('Test_Set.csv')
    training_set = dataset_train.iloc[:,indx:(indx+1)].values
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    y_train = []
    prev_data = 30
    for i in range(prev_data, training_set.shape[0]):
        X_train.append(training_set_scaled[i-prev_data:i, 0])
        y_train.append(training_set_scaled[i, 0])	
    X_train, y_train = np.array(X_train),np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
   
    regressor = Sequential()
    #First LSTM layer and dropout regularisation
    regressor.add(LSTM(units=70, return_sequences=True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    #Adding 2nd LSTM and dropout regularisation
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.2))
    #Adding 3rd LSTM and dropout regularisation
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.2))
    #Adding 4th LSTM and dropout regularisation
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.2))
    #Adding 5th LSTM and dropout regularisation
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.2))
    #Adding 6th LSTM and dropout regularisation
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.2))
    #Adding 7th LSTM and dropout regularisation
    regressor.add(LSTM(units=70))
    regressor.add(Dropout(0.2))
    #Output layer
    regressor.add(Dense(units = 1))
    #Compiling RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    #Fitting RNN to Training set
    regressor.fit(X_train, y_train, epochs = 2000, batch_size = 24)
    dataset_test = pd.read_csv('Test_Set.csv')
    dataset_total = pd.concat((dataset_train[prd_num], dataset_test[prd_num]), axis = 0)
    inputs = dataset_total[len(dataset_total)-len(dataset_test) - prev_data:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(30, 59):
        X_test.append(inputs[i-30:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], 30, 1))
    predicted_number = regressor.predict(X_test)
    predicted_number = sc.inverse_transform(predicted_number)
    return predicted_number


data_p = []


for k in range(1,100):
    prd_num = str(product_ids[k-1][0])
    data_p.append(predictFunc(k, prd_num))
    
#dataset_train1 = pd.read_csv('Training_Set_Full.csv') 
#training_set1 = dataset_train1.iloc[:,k:(k+1)].values

#plt.plot(data_p[i], color ='red', label = 'Predicted data')
#plt.show()
#plt.savefig('test.png')




	

	
	
	
	
	
	
	
	
	
	