#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 21:45:42 2021

@author: mariapalacios
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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
print('Classification summary:')
print(classification_report(y, est.predict(X)))
# print('Classification summary:')
# print(classification_report(y, est.predict(X)))

# El clasificador base solo usa age, avg_glucose_level, bmi
# Busca explicaciones en las que aparezcan variables distintas a estas tres.
# Usa X e y como datos (los nombres de las columnas estan en X_features), est  es el clasificador final


from sklearn.metrics import classification_report
import numpy as np
#import dalex as dx
import shap

i=2500

X=np.array(X)
#SHAP EXPLAINER
explainer = shap.LinearExplainer(est, X)
shap_values = explainer.shap_values(X)# Estima los valores de shaply en el conjunto de datos de prueba
#explicaciones globales
shap.summary_plot(shap_values, X, feature_names=X_features)
#Lo de forzar es para casos indivuduales
shap.force_plot(explainer.expected_value,shap_values[i,:], X[i,:], feature_names=X_features, matplotlib=True)


##shap.dependence_plot("age", shap_values, X)
##shap.dependence_plot("avg_glucose_level", shap_values, X)
##shap.dependence_plot("BMI", shap_values, X)



##DALEX EXPLAINER???
#exp_gbm = dx.Explainer(est, data=X, y=y, predict_function=est.predict(X), label='gbm')
#pp = exp_gbm.predict_parts(X.iloc[[i]], type='shap_wrapper', shap_explainer_type="TreeExplainer")
#type(pp)
#pp.plot()
#pp.result  # shap_values
#mp = exp_gbm.model_parts(type='shap_wrapper', shap_explainer_type="TreeExplainer")
#type(mp)
#mp.plot()
#mp.plot(plot_type='bar')
#mp.result  # shap_values