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


import shap
class_names=['healthy','stroke']
#print(X[i].mean)
X=np.array(X)
X_features=np.array(X_features)
#SHAP EXPLAINER
#explainer = shap.LinearExplainer(est, X)#lo que utilizamos al principio

import pickle
with open('explainer.pkl', 'rb') as fd:
    explainer = pickle.load(fd)
with open('shap_values.pkl', 'rb') as fd:
    shap_values = pickle.load(fd)

# rather than use the whole training set to estimate expected values, we summarize with
# a set of weighted kmeans, each weighted by the number of points they represent.
#explainer = shap.KernelExplainer(est.predict, shap.kmeans(X, 100))
#shap_values = explainer.shap_values(X)# Estima los valores de shaply en el conjunto de datos de prueba
    
# we can use shap.approximate_interactions to guess which features
# may interact with age
#inds_age = shap.approximate_interactions(20, shap_values, X)
#inds_avg_glucose_level = shap.approximate_interactions(21, shap_values, X)
#inds_bmi = shap.approximate_interactions(22, shap_values, X)
#shap.summary_plot(shap_values, X, feature_names=X_features)
#for i in range(2):
#    shap.dependence_plot(20, shap_values, X, feature_names=X_features, interaction_index=inds_age[i])#age
#    
#shap.dependence_plot(21, shap_values, X, feature_names=X_features, interaction_index=22)#average glucose level
#shap.dependence_plot(22, shap_values, X, feature_names=X_features, interaction_index=20)#bmi
##shap.decision_plot(explainer.expected_value, shap_values, X_features, ignore_warnings=True)



correct_explanation=0
incorrect_explanation=0

correct_prediction=True
counter_correct_predictions=0
counter_incorrect_predictions=0
correct_prediction_correct_explanation=0
correct_prediction_incorrect_explanation=0
incorrect_prediction_correct_explanation=0
incorrect_prediction_incorrect_explanation=0
    
#i=666

for i in range(len(dataset)):
    tmp2=np.sum(shap_values[i]!= 0)
    tmp=np.sum(shap_values[i][20:23] != 0)
    if (est.predict_proba([X[i]])[0, 1] ) >= 0.5:
        prediction='stroke'
    else:
        prediction='healthy'
        
    if (prediction=='healthy') and (class_names[y[i]]=='healthy'):
        correct_prediction=True
        counter_correct_predictions=counter_correct_predictions+1
        if tmp2 == 3 and tmp ==3:
            correct_prediction_correct_explanation+=1
            correct_explanation+=1
        else:
            correct_prediction_incorrect_explanation+=1   
            incorrect_explanation+=1
    elif (prediction=='stroke') and (class_names[y[i]]=='stroke'):
        counter_correct_predictions=counter_correct_predictions+1
        correct_prediction=True
        if tmp2 == 3 and tmp ==3:
            correct_prediction_correct_explanation+=1
            correct_explanation+=1
        else:
            correct_prediction_incorrect_explanation+=1  
            incorrect_explanation+=1
    else:
        correct_prediction=False
        counter_incorrect_predictions=counter_incorrect_predictions+1
        if tmp2 == 3 and tmp ==3:
            incorrect_prediction_correct_explanation+=1
            correct_explanation+=1
        else:
            incorrect_prediction_incorrect_explanation+=1  
            incorrect_explanation+=1
        
    
       
    print("*************************************************************")
    print('Document id: %d' % i)
    print('Probability(stroke) =', est.predict_proba([X[i]])[0, 1])
    print('True class: %s' % class_names[y[i]])
    print("*************************************************************")
    #Lo de forzar es para casos individuales
    #shap.force_plot(explainer.expected_value,shap_values[i,:], X[i,:], feature_names=X_features, matplotlib=True)
    #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[i,:], X[i,:], feature_names=X_features)
        
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
    
        
print('all')
percentage_total_correct_explanations=correct_explanation/len(dataset)
#percentage_total_correct_columns=correct_columns/(len(dataset)*3)
print(percentage_total_correct_explanations)   
#print(percentage_total_correct_columns) 
print('correct prediction')
print(correct_prediction_correct_explanation/counter_correct_predictions)
#print(counter_columns_correct_predictions/(counter_correct_predictions*3))
print('incorrect prediction')
print(incorrect_prediction_correct_explanation/counter_incorrect_predictions)
#print(counter_columns_incorrect_predictions/(counter_incorrect_predictions*3))

