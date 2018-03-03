
import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.ensemble import RandomForestClassifier
import pydotplus

dataset = pd.read_csv('/Users/danny/Desktop/Practice Deep Learning/bank/bank.csv',sep=';',names=['age','job','martial','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'])
dataset.replace(('unknown'),('0'),inplace=True)
dataset.replace(('married','divorced','single'),(1,2,3),inplace=True)
dataset.replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed'),(1,2,3,4,5,6,7,8,9,10,11),inplace=True)
dataset.replace(('primary','secondary','tertiary'),(1,2,3),inplace=True)
dataset.replace(('cellular','telephone'),(1,2),inplace=True)
dataset.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(0,1,2,3,4,5,6,7,8,9,10,11),inplace=True)
dataset.replace(('failure','nonexistent','success','other'),(1,2,3,4),inplace=True)
dataset.replace(('yes','no'),(1,0), inplace=True)

features = list(dataset.columns[:16])
y = dataset['y']
X = dataset[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, y)



print(clf.predict([[30,11,1,1,0,1787,0,0,1,19,9,79,1,1,0,0]]))
print(clf.predict([[68,6,2,2,0,4189,0,0,2,14,6,897,2,1,0,0]]))

