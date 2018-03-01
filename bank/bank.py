import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import cross_val_score

feature_names = ['age','job','martial','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y']
dataset = pd.read_csv('/Users/danny/Desktop/Practice Deep Learning/bank/bank.csv',sep=';',names = feature_names)
dataset.dropna(inplace=True)
dataset.replace(('yes','no'),(1,0), inplace=True)
dataset.replace(('unknown'),('0'),inplace=True)
dataset.replace(('married','divorced','single'),(1,2,3),inplace=True)
dataset.replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed'),(1,2,3,4,5,6,7,8,9,10,11),inplace=True)
dataset.replace(('primary','secondary','tertiary'),(1,2,3),inplace=True)
dataset.replace(('cellular','telephone'),(1,2),inplace=True)
dataset.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(0,1,2,3,4,5,6,7,8,9,10,11),inplace=True)
dataset.replace(('failure','nonexistent','success','other'),(1,2,3,4),inplace=True)

all_features = dataset[feature_names].drop('y',axis=1).values
all_classes = dataset['y'].values

def create_model():
    model = Sequential()
    model.add(Dense(32, input_dim=16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
    
estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, verbose=1)
cv_scores = cross_val_score(estimator, all_features, all_classes, cv=10)
cv_scores.mean()

