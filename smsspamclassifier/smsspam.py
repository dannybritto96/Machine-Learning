import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
f = open('SMSSpamCollection',"r")

# Data Preprocessing
y = []
x = []
z = []
for line in f:
    temp = line
    line = re.sub('ham\t','',line)
    line = re.sub('spam\t','',line)
    line = line.rstrip()
    y.append(line)
    x.append(('\n'.join(item.split()[0] for item in temp.split('\t'))))

for i in range(len(x)):
    if x[i].startswith('ham'):
        z.append(x[i].split('\n')[0])
    if x[i].startswith('spam'):
        z.append(x[i].split('\n')[0])
del(x)

# Loading Data into Dataframe

dataset = pd.DataFrame({'classification':z,
                        'message':y})

# Implementing Count Vectorizer and MultinomialNB

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(dataset['message'].values)

classifier = MultinomialNB()
targets = dataset['classification'].values
classifier.fit(counts,targets)

# Make new predictions

testdata = ["Free money offer!!","Hey let's meet over a cup of coffee"]
testdata_counts = vectorizer.transform(testdata)
predictions = classifier.predict(testdata_counts)
print(predictions)
