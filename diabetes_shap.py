#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:28:01 2021

@author: mariapalacios
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

RANDOM_SEED=42

# No uses rutas absolutas (solo funcionan en tu ordenador), usa rutas relativas
dataset = pd.read_csv("datasets/diabetes2.csv")
dataset.head()

X = dataset.iloc[:, :-1]
X_features = X.columns
y = dataset.iloc[:, -1].values

scaler=StandardScaler()
X = scaler.fit_transform(X)

C=0.015
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

class_names=['no_diabetes','diabetes']
X=np.array(X)
X_features=np.array(X_features)

#explainer = shap.LinearExplainer(est, X)#lo que utilizamos al principio
#explainer = shap.KernelExplainer(est.predict, shap.kmeans(X, 100))
#shap_values = explainer.shap_values(X)# Estima los valores de shaply en el conjunto de datos de prueba

import pickle
with open('diabetes-shap.pkl', 'rb') as fd:
    [explainer, shap_values] = pickle.load(fd)

shap.summary_plot(shap_values, X, feature_names=X_features)
#shap.decision_plot(explainer.expected_value, shap_values, X_features, ignore_warnings=True)

correct_explanation=0
incorrect_explanation=0

correct_prediction=True
counter_correct_predictions=0
counter_incorrect_predictions=0


for i in range(len(dataset)):  
    if (est.predict_proba([X[i]])[0, 1] ) >= 0.5:
            prediction='diabetes'
    else:
            prediction='no_diabetes'
            
    if (prediction=='no_diabetes') and (class_names[y[i]]=='no_diabetes'):
        correct_prediction=True
        counter_correct_predictions=counter_correct_predictions+1
    elif (prediction=='diabetes') and (class_names[y[i]]=='diabetes'):
        counter_correct_predictions=counter_correct_predictions+1
        correct_prediction=True
    else:
        correct_prediction=False
        counter_incorrect_predictions=counter_incorrect_predictions+1
        
    tmp2=np.sum(shap_values[i]!= 0)
    tmp=np.sum(shap_values[i][0:2] != 0)+np.sum(shap_values[i][5] != 0)
    if tmp2 == 3 and tmp ==3:
        correct_explanation+=1
    elif tmp2 > 3:
        incorrect_explanation+=1      
        print("*************************************************************")
        print('Document id: %d' % i)
        print('Probability(diabetes) =', est.predict_proba([X[i]])[0, 1])
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