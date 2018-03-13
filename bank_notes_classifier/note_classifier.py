# http://archive.ics.uci.edu/ml/datasets/banknote+authentication

'''
Attribute Information:

1. variance of Wavelet Transformed image (continuous) 
2. skewness of Wavelet Transformed image (continuous) 
3. curtosis of Wavelet Transformed image (continuous) 
4. entropy of image (continuous) 
5. class (integer) 

Data were extracted from images that were taken for the evaluation of an authentication procedure for banknotes.

Data were extracted from images that were taken from genuine and forged banknote-like specimens. 
For digitization, an industrial camera usually used for print inspection was used. 
The final images have 400x 400 pixels. 
Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. 
Wavelet Transform tool were used to extract features from images.


'''

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np

# Loading data into DataFrame
dataset = pd.read_csv("data_banknote_authentication.txt",names=['variance','skewness','curtosis','entropy','class'])
# Shuffling data for Train Test sampling
dataset = dataset.sample(frac=1).reset_index(drop=True)

X = dataset.iloc[:, :4]
y = dataset.iloc[:, 4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()
classifier.add(Dense(units = 8,kernel_initializer='random_uniform',activation='relu',input_dim = 4))
classifier.add(Dense(units = 8,kernel_initializer='random_uniform',activation='relu' ))
classifier.add(Dense(units = 4,kernel_initializer='random_uniform',activation='relu'))
classifier.add(Dense(units = 1,kernel_initializer='random_uniform',activation='sigmoid'))
classifier.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train, batch_size=20, epochs=100)

# Test data
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Make new prediction

new_prediction = classifier.predict(sc.transform(np.array([[4.9294,0.27726999999999996,0.20792,0.33662]])))
new_prediction = (new_prediction > 0.5)
