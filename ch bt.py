import re

import numpy as np
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
df = pd.read_csv("C:/Users/Star World/Desktop/dialogs.txt",sep='\t')
a = pd.Series(df.columns)
a = a.rename({0: df.columns[0],1: df.columns[1]})
b = {'Questions':'Hi','Answers':'hello'}

c = {'Questions':'Hello','Answers':'hi'}

d= {'Questions':'how are you','Answers':"i'm fine. how about yourself?"}

e= {'Questions':'how are you doing','Answers':"i'm fine. how about yourself?"}


df = pd.concat([df, pd.DataFrame([a])], ignore_index=True)
df.columns=['Questions','Answers']

df = pd.concat([df, pd.DataFrame([b, c, d,e])], ignore_index=True)
def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]
Pipe = Pipeline([
    ('bow',CountVectorizer(analyzer=cleaner)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])
Pipe.fit(df['Questions'],df['Answers'])
def val():
    qu=input("please enter your question")
    if re.findall("quit",qu):
        print("thanks")
    else:
        me=Pipe.predict([qu])[0]
        print(me)
        val()
val()