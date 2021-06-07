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

i=556
class_names=['no_diabetes','diabetes']
X=np.array(X)
X_features=np.array(X_features)

#SHAP EXPLAINER
explainer = shap.LinearExplainer(est, X)
shap_values = explainer.shap_values(X)# Estima los valores de shaply en el conjunto de datos de prueba
#explicaciones globales
#shap.summary_plot(shap_values, X, feature_names=X_features)
#Lo de forzar es para casos individuales
print("*************************************************************")
print('Document id: %d' % i)
print('Probability(diabetes) =', est.predict_proba([X[i]])[0, 1])
print('True class: %s' % class_names[y[i]])
print("*************************************************************")
shap.force_plot(explainer.expected_value,shap_values[i,:], X[i,:], feature_names=X_features, matplotlib=True)
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[i,:], X[i,:], feature_names=X_features)
#shap.decision_plot(explainer.expected_value, shap_values, X_features, ignore_warnings=True)
