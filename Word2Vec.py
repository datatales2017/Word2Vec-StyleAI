# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 13:01:06 2017

@author: manasa
"""

import pandas as pd
import nltk
nltk.download('punkt')
import gensim
from gensim import corpora,models,similarities
import re

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('jokes.csv')

X =  (df['Question']).tolist()
y = (df['Answer']).tolist()

corpus = (X+y)
corpus1 = []
tok_test = []
for i in range(len(corpus)):
    test = re.sub('^A-Z a-z',' ',corpus[i])
    test = re.sub(r'[^\w]', ' ', corpus[i])
    test = test.lower()
    test=test.split()
    test = ' '.join(test)
    corpus1.append(test)
    tok_test.append(nltk.word_tokenize(corpus[i]))
    
model = gensim.models.Word2Vec(tok_test,min_count=1,size=32)
#model.save('StyleAI') #for saving
#model = gensim.models.Word2Vec.load('StyleAI') #for loading

from sklearn.externals import joblib
filename = 'finalized_model.pkl'
joblib.dump(model, filename)
loaded_model = joblib.load(filename)
result = loaded_model.most_similar('red')
corr = pd.DataFrame(result,columns ={'item','correlation'})
corr.index = corr.index + 1
print(corr)