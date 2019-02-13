# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:18:11 2018

@author: Akash
"""

# ==========================================================================================================================================================
#                                 O B J E C T I V E -  Predict the opening price of Google Stock at any day
# ==========================================================================================================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =================================================================================================================================================
#                                           P A R T - 1       D A T A   P R E P R O C E S S I N G
# ==========================================================================================================================================================


# -------Dataset - Dataset is a csv file of Google Stock prices from 2012-2016 with Open-Close-High-low value

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values                 # Extracting only the OPEN value from the dataset and converting into array


# -------Feature Scaling-----------
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set_scaled = sc.fit_transform(training_set)

# =============================================================================
#  Creating a flow of 60 timestamp and 1 output
# 
#  It means for every 1 output 60 values will be considered. Means to predict the opening price of a particular day last 60days opening price is 
#  trained and analysied to predict todays price
# =============================================================================

X_train = []
Y_train = []

for i in range(60,1258):        #range(from,to)
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])

X_train,Y_train = np.array(X_train),np.array(Y_train)
    

# ---------Reshaping-----------
#        Reshaping to change the dimension of the array from 2D to 3D. 
#        Reshaping can be done if we want to add more then 1 indicator to our model
#        Indicator - Values that may help in better predecting our model

X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
# =============================================================================
#         Reshape --   (array to be reshaped, newshape)
#                     Array to be reshaped - X_train
#                     Newshape - (batchsize,timestamp(=60),no. of indicators)
# =============================================================================


# ==========================================================================================================================================================
#                                       P A R T - 2    Initializing the R.N.N
# ==========================================================================================================================================================

# --------Libraries-----------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# --------Initialize the RNN----------
regressor = Sequential()


# -------- Adding 1st LSTM Layers and some Dropout Regularisation------------------
regressor.add(LSTM(units= 50,return_sequences=True,input_shape=(X_train.shape[1],1)))
#                units = no. of Neural Networks used.
#                input_shape = requires only last 2 arguments,batchsize is taken automatically

regressor.add(Dropout(0.2))             # 20% Dropout

# -------- Adding 2nd LSTM Layers and some Dropout Regularisation------------------
regressor.add(LSTM(units= 50,return_sequences=True))
regressor.add(Dropout(0.2))

# -------- Adding 3rd LSTM Layers and some Dropout Regularisation------------------
regressor.add(LSTM(units= 50,return_sequences=True))
regressor.add(Dropout(0.2))

# -------- Adding 4th LSTM Layers and some Dropout Regularisation------------------
regressor.add(LSTM(units= 50))
regressor.add(Dropout(0.2))
                
# -------- Adding the output Layer ---------------------
regressor.add(Dense(units=1))

# -------- Compiling the RNN --------------
regressor.compile(optimizer = 'adam',loss='mean_squared_error')

# -------- Training the RNN model -----------
regressor.fit(X_train,Y_train,epochs = 100, batch_size= 32)


# ==========================================================================================================================================================
#                       P A R T -3      P R E D I C T I O N   &   V I S U A L I S A T I O N 
# ==========================================================================================================================================================

# -------- Importing Test Data --------------

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# ---------Preprocessing Test Data------------
#            2017 Google Stock Data            

# Combining train+test data
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#                    We want data of dates 60 days back from 1st working day of January 2017

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

for i in range(60,80):        #range(from,to)
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

# -------- P r e d i c t i o n ---------
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# ---------- V I S U A L I Z A T I O N -----------

plt.plot(real_stock_price,color = 'red',label='Real Stock Price')
plt.plot(predicted_stock_price,color = 'blue',label = 'Predicted Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()






