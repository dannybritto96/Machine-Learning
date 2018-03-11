import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('/Users/danny/Desktop/DeepLearning/mammographic_masses.data.txt',na_values=['?'],names = ['BI-RADS', 'age', 'shape', 'margin', 'density', 'severity'])
dataset.dropna(inplace=True)
all_features = dataset[['age','shape','margin','density']].values
all_classes = dataset['severity'].values

scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_features)

##############
# Grid Classifier to find out the best paramters within the paramters dictionary!
#def build_classifier(optimizer):
#    model = Sequential()
#    model.add(Dense(units = 6,kernel_initializer = 'normal', activation='relu',input_dim=4))
#    model.add(Dense(units = 6,kernel_initializer = 'normal', activation='relu'))
#    model.add(Dense(units = 4,kernel_initializer = 'normal', activation='relu'))
#    model.add(Dense(units = 1,kernel_initializer = 'normal', activation='sigmoid'))
#    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
#    return model
#
#classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size':[25,30],
#              'epochs':[100,300],
#              'optimizer':['adam','rmsprop']}
#
#grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',cv = 10)
#grid_search = grid_search.fit(scaled_features,all_classes)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_

################

model =Sequential()
model.add(Dense(units = 6,kernel_initializer = 'normal', activation='relu',input_dim=4))
model.add(Dense(units = 6,kernel_initializer = 'normal', activation='relu'))
model.add(Dense(units = 4,kernel_initializer = 'normal', activation='relu'))
model.add(Dense(units = 1,kernel_initializer = 'normal', activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(scaled_features,all_classes,batch_size=30,epochs = 300)

#################
# Make new predictions
new_prediction = model.predict(scaler.transform(np.array([[42,3,2,3]])))
new_prediction = (new_prediction > 0.5)
