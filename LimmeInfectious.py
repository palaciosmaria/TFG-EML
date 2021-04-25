#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:12:54 2021

@author: mariapalacios
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import pandas as pd

train= pd.read_csv("/Users/mariapalacios/Desktop/TFG/datasets/INFECTIOUS_DISEASE_PREDICTION/train.csv")
test= pd.read_csv("/Users/mariapalacios/Desktop/TFG/datasets/INFECTIOUS_DISEASE_PREDICTION/test.csv")


#X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 1/3, random_state = 0)

class_names = ['1', '0']


rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train, y_train)
print(rf)

pred = rf.predict(X_test)
print('Accuracy: {}'.format(sklearn.metrics.accuracy_score(y_test, pred)))

from lime import lime_text
from sklearn.pipeline import make_pipeline

print(rf.predict_proba([X_test.data[0]]))

#Now we create an explainer object. We pass the class_names a an argument for prettier display.
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

#We then generate an explanation with at most 6 features for an arbitrary document in the test set.
idx = 83
exp = explainer.explain_instance(X_test.data[idx], rf.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(christian) =', rf.predict_proba([X_test.data[idx]])[0,1])
print('True class: %s' % class_names[X_test.target[idx]])

#The classifier got this example right (it predicted atheism).
#The explanation is presented below as a list of weighted features.
exp.as_list()

#These weighted features are a linear model, which approximates the behaviour of the random forest classifier 
#in the vicinity of the test example. Roughly, if we remove 'Posting' and 'Host' from the document , 
#the prediction should move towards the opposite class (Christianity) by about 0.27 
#(the sum of the weights for both features). Let's see if this is the case.
print('Original prediction:', rf.predict_proba(y_test[idx])[0,1])
tmp = y_test[idx].copy()
#tmp[0,vectorizer.vocabulary_['Posting']] = 0
#tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(y_test[idx])[0,1])

#Pretty close!
#The words that explain the model around this document seem very arbitrary - not much to do with 
#either Christianity or Atheism.
#In fact, these are words that appear in the email headers (you will see this clearly soon), 

#VISUALIZING EXPLANATIONS
#%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/Users/mariapalacios/Desktop/TFG/oi.html')
exp.show_in_notebook(text=True)
