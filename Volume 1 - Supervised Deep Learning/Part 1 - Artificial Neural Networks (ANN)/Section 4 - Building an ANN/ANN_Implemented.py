import pandas as pd
import numpy as np
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values
#encoding the data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])
labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features =[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train  = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
#initialising the ANN
classifier= Sequential()
classifier.add(Dense(output_dim =6,init ='uniform',activation ='relu',input_dim=11))
#Adding the second hidden layer
classifier.add(Dense(output_dim =6,init ='uniform',activation ='relu'))
#Adding the output layer
classifier.add(Dense(output_dim =1,init ='uniform',activation ='sigmoid'))
#Compile the ANN
classifier.compile(optimizer='adam',loss= 'binary_crossentropy',metrics = ['accuracy'])
#fitting the neural network to datset
classifier.fit(X_train,Y_train,batch_size=10,epochs =100)
#making the prediction
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred>0.5)
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
