# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:50:59 2018

@author: Akash
"""

# ==========================================================================================================================================================
#                                   O B J E C T I V E  ---  To predict if a bank customer will leave the bank or not
# ==========================================================================================================================================================


# =============================================================================
#                                       PART -- 1   Data Preprocessing
# =============================================================================

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]         # To avoid Dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
#                                        PART - 2   Building ANN Model
# =============================================================================
import keras
from keras.models import Sequential       # Used to initialize the Neural Network
from keras.layers import Dense            # Used to build layers of the ANN model

# Initialize the ANN object
classifier = Sequential()       #Classifier - as we are predicting a Yes/No condition

#   Adding the Input Layer and First Hidden Layer

# classifier.add(Dense(output_dim = 6,init = "uniform",activation = 'relu', input_dim = 11))   As in Tutorial, but now the parameter has changed
classifier.add(Dense(activation='relu', input_dim=11, units = 6, kernel_initializer='uniform'))

# =============================================================================
# activation = 'relu'    Defines to use Rectifier function for training Neural network
# 
# input_dim = 11         Denifes the no. of input layers to be 11.
# 
# units/output_dim = 6   The number of hidden layers to be 6. It's usually difficult to decide how many Hidden layer to keep.
#                         So, we use a formula as - (input_Layer + output_Layer)/2 = Hidden Layer
#                         
# Kernel_initializer/init = 'uniform'     Telling model to assign weight of each input layer uniformly
# =============================================================================

#    Adding the Second Hidden layer
classifier.add(Dense(activation="relu", units = 6, kernel_initializer='uniform'))

#    Adding the Output Layer
classifier.add(Dense(activation="sigmoid", units = 1, kernel_initializer='uniform'))
#                                In the output layer we are using sigmoid so that we can get a probalistic output to if the customer will leave the bank or not.
        
#   Note ---
#       If in some case, we have more then 2 output categories, then we use the kernel_initializer = "softmax"

#    Compiling the ANN
classifier.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
#                    Optimizer = The algorithm that we are going to use to find the optimal set of weights in the neural network
#                                and we are going to use 'Sochastic Gradient Descent' for that and specifically 'adam'(Sochastic Gradient Descent)
#                                
#                    loss =  The loss function we are going to use to optimize the weight of Neural Network.
                    
#    Fitting the dataset to the Model and training the model
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
#                    Batch-Size = The batch of sample after which we want to update our weight.
#                    Epochs =    How many iterations we want to perform to train the model


# =============================================================================
#                                         PART - 3    Predicting and Evaluating
# =============================================================================
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# =============================================================================
#                                 Model Accuracy = 86%
# =============================================================================



# =============================================================================
#                     Predicting it for a new Value
# =============================================================================

#Geography: France
#Credit Score: 600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#Does this customer have a credit card ? Yes
#Is this customer an Active Member: Yes
#Estimated Salary: $50000


new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5)

" Since new_pred = False, It means that the customer will not leave the bank"


# =============================================================================
#                 Checking accuracy of model using K-Fold Cross Validation
# =============================================================================

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential() 
    classifier.add(Dense(activation='relu', input_dim=11, units = 6, kernel_initializer='uniform'))
    classifier.add(Dense(activation="relu", units = 6, kernel_initializer='uniform'))
    classifier.add(Dense(activation="sigmoid", units = 1, kernel_initializer='uniform'))
    classifier.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

new_classifier = KerasClassifier(build_fn = build_classifier,batch_size=10,epochs=100)
accuracies = cross_val_score(estimator = classifier,X=X_train,Y=y_train,cv=10,n_jobs=-1)
mean =  accuracies.mean()
variance = accuracies.std()



#    Accuracies tell how the model gives different accuracies over 10 different folds of training set

# =============================================================================
#                                            Tuning the ANN
#     
#     In this we use GridSearchCV to tune the ANN for different parameters value.
#     In Grid_search, the model is trained for different combinations of the parameter values and at last best result is obtained from all possible
#     combination of parameters value.
#      Here we use multiple epoch,batch-size for differnt optimizers.
# =============================================================================
     
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV            
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential() 
    classifier.add(Dense(activation='relu', input_dim=11, units = 6, kernel_initializer='uniform'))
    classifier.add(Dense(activation="relu", units = 6, kernel_initializer='uniform'))
    classifier.add(Dense(activation="sigmoid", units = 1, kernel_initializer='uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameter = {'batch_size':[25,32],
             'nb_epoch':[100,500],
             'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameter,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fir(X_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_












