#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:05:43 2021

@author: mariapalacios
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report

RANDOM_SEED=42

# No uses rutas absolutas (solo funcionan en tu ordenador), usa rutas relativas
dataset = pd.read_csv("datasets/healthcare-dataset-stroke-data.csv")
dataset.head()

# Let's give better names to some categorical levels
dataset['hypertension'] = dataset['hypertension'].replace([0, 1], ['healthy-tension', 'hypertense'])
dataset['heart_disease'] = dataset['heart_disease'].replace([0, 1], ['healthy-heart', 'heart-disease'])
dataset['ever_married'] = dataset['ever_married'].replace(['No', 'Yes'], ['not-ever-married', 'ever-married'])
dataset['smoking_status'] = dataset['smoking_status'].replace(['Unknown'], ['unknown'])

X = dataset.drop(['id'], axis=1).iloc[:, :-1]
y = dataset.iloc[:, -1].values
# print(np.unique(y, return_counts=True)) # count number of 0s and 1s

# Remove 'Other' gender for easining the interpretation of results
dataset = dataset.loc[dataset['gender'] != 'Other', :]

# Careful, this should be reviewed for each dataset
numerical_idx = X.select_dtypes(include=['float64']).columns
categorical_idx = X.select_dtypes(include=['object', 'bool', 'int64']).columns
assert X.shape[1] == (len(numerical_idx) + len(categorical_idx)), 'some column is missing'

# impute BMI's NAs
# X.columns[X.isna().any()].tolist() # This line may be used to check that indeed BMI has NAs
numerical_imp = SimpleImputer()
X_numerical = numerical_imp.fit_transform(X[numerical_idx])

# Encode categorical variables
oh_enc = OneHotEncoder()
X_cat = oh_enc.fit_transform(X[categorical_idx]).todense()

X = np.concatenate([X_cat, X_numerical], axis=1)
X_features = np.concatenate([oh_enc.get_feature_names(), numerical_idx])


C = 0.001
est = LogisticRegression(penalty='l1', C=C, random_state=RANDOM_SEED, solver='saga', max_iter=1000,
                         class_weight='balanced')
est.fit(X, y)
coefs = est.coef_.ravel()
relevant_columns = X_features[coefs != 0]
sparsity_l1 = np.mean(coefs == 0) * 100
print(f"C={C}")
print(f"Sparsity with L1 penalty: {sparsity_l1} ({np.sum(coefs != 0)} / {len(coefs)} columns)")
print(f"Relevant columns are: {', '.join(relevant_columns)}")
# print('Classification summary:')
# print(classification_report(y, est.predict(X)))

# El clasificador base solo usa age, avg_glucose_level, bmi
# Busca explicaciones en las que aparezcan variables distintas a estas tres.
# Usa X e y como datos (los nombres de las columnas estan en X_features), est  es el clasificador final


class_names = ['1', '0']
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)

#We then generate an explanation with at most 6 features for an arbitrary document in the test set.
idx = 83
exp = explainer.explain_instance(X_features.data[idx], est.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(stroke) =', est.predict_proba([X.data[idx]])[0,1])
print('True class: %s' % class_names[X.target[idx]])

#The classifier got this example right (it predicted atheism).
#The explanation is presented below as a list of weighted features.
exp.as_list()


print('Original prediction:', est.predict_proba(y[idx])[0,1])
tmp = y[idx].copy()

print('Prediction removing some features:', est.predict_proba(tmp)[0,1])
print('Difference:', est.predict_proba(tmp)[0,1] - est.predict_proba(y[idx])[0,1])


#VISUALIZING EXPLANATIONS
#%matplotlib inline
fig = exp.as_pyplot_figure()
exp.show_in_notebook(text=False)
exp.save_to_file('/Users/mariapalacios/Desktop/TFG/oi.html')
exp.show_in_notebook(text=True)






