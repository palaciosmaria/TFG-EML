#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:26:07 2021

@author: mariapalacios
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
RANDOM_SEED=42

# No uses rutas absolutas (solo funcionan en tu ordenador), usa rutas relativas
dataset = pd.read_csv("datasets/heart_failure_clinical_records_dataset.csv")
dataset.head()

dataset['anaemia'] = dataset['anaemia'].replace([0, 1], ['not-anaemia', 'anaemia'])
dataset['diabetes'] = dataset['diabetes'].replace([0, 1], ['not-diabetes', 'diabetes'])
dataset['high_blood_pressure'] = dataset['high_blood_pressure'].replace([0, 1], ['normal-bp', 'high-bp'])
dataset['sex'] = dataset['sex'].replace([0, 1], ['Female', 'Male'])
dataset['smoking'] = dataset['smoking'].replace([0, 1], ['not-smoker', 'smoker'])


X = dataset.drop('time', axis=1).iloc[:, :-1]
X_features = X.columns
y = dataset.iloc[:, -1].values

numerical_idx = X.select_dtypes(include=['float64', 'int64']).columns
categorical_idx = X.select_dtypes(include=['object', 'bool']).columns
assert X.shape[1] == (len(numerical_idx) + len(categorical_idx)), 'some column is missing'

scaler=StandardScaler()

# Encode categorical variables
X_numerical = scaler.fit_transform(X[numerical_idx])
oh_enc = OneHotEncoder()
X_cat = oh_enc.fit_transform(X[categorical_idx]).todense()


X = np.concatenate([X_cat, X_numerical], axis=1)
X_features = np.concatenate([oh_enc.get_feature_names(), numerical_idx])

C = 0.025
est = LogisticRegression(penalty='l1', C=C, random_state=RANDOM_SEED, solver='saga', max_iter=1000,
                         class_weight='balanced')
est.fit(X, y)
coefs = est.coef_.ravel()
relevant_columns = X_features[coefs != 0]
sparsity_l1 = np.mean(coefs == 0) * 100
print(f"C={C}")
print(f"Sparsity with L1 penalty: {sparsity_l1} ({np.sum(coefs != 0)} / {len(coefs)} columns)")
print(f"Relevant columns are: {', '.join(relevant_columns)}")
print('Classification summary:')
print(classification_report(y, est.predict(X)))


import shap
class_names=['healthy','heart_failure']
X=np.array(X)
X_features=np.array(X_features)
#SHAP EXPLAINER
#explainer = shap.LinearExplainer(est, X)

# rather than use the whole training set to estimate expected values, we summarize with
# a set of weighted kmeans, each weighted by the number of points they represent.
#explainer = shap.KernelExplainer(est.predict, shap.kmeans(X, 100))
#shap_values = explainer.shap_values(X)# Estima los valores de shaply en el conjunto de datos de prueba

import pickle
with open('heart_failure-shap.pkl', 'rb') as fd:
    [explainer, shap_values] = pickle.load(fd)

shap.summary_plot(shap_values, X, feature_names=X_features)
#shap.decision_plot(explainer.expected_value, shap_values, X_features, ignore_warnings=True)

correct_explanation=0
incorrect_explanation=0

correct_prediction=True
counter_correct_predictions=0
counter_incorrect_predictions=0

i=22
for i in range(len(dataset)):
    if (est.predict_proba([X[i]])[0, 1] ) >= 0.5:
            prediction='heart_failure'
    else:
            prediction='healthy'
            
    if (prediction=='healthy') and (class_names[y[i]]=='healthy'):
        correct_prediction=True
        counter_correct_predictions=counter_correct_predictions+1
    elif (prediction=='heart_failure') and (class_names[y[i]]=='heart_failure'):
        counter_correct_predictions=counter_correct_predictions+1
        correct_prediction=True
    else:
        correct_prediction=False
        counter_incorrect_predictions=counter_incorrect_predictions+1
            
    tmp2=np.sum(shap_values[i]!= 0)
    tmp=np.sum(shap_values[i][12] != 0)+np.sum(shap_values[i][14] != 0)
    if tmp2 == 2 and tmp ==2:
        correct_explanation+=1
    elif tmp2 > 2:
        incorrect_explanation+=1      
        print("*************************************************************")
        print('Document id: %d' % i)
        print('Probability(heart_failure) =', est.predict_proba([X[i]])[0, 1])
        print('True class: %s' % class_names[y[i]])
        print("*************************************************************")
        #Lo de forzar es para casos individuales
        #shap.force_plot(explainer.expected_value,shap_values[i,:], X[i,:], feature_names=X_features, matplotlib=True)
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[i,:], X[i,:], feature_names=X_features)
            
        
print('correct prediction')
print(counter_correct_predictions/len(dataset))
print('incorrect prediction')
print(counter_incorrect_predictions/len(dataset))

percentage_total_correct_explanations=correct_explanation/len(dataset)
percentage_total_incorrect_explanations=incorrect_explanation/len(dataset)

print('correct explanations')
print(percentage_total_correct_explanations)
print('incorrect explanations')
print(percentage_total_incorrect_explanations)