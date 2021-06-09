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

import lime.lime_tabular
#from lime.lime_text import LimeTabularExplainer
class_names=['healthy','stroke']
X= np.asarray(X)
explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names = list(X_features), 
                                                  class_names=class_names,
                                                  categorical_features=range(X_cat.shape[1]),
                                                  discretize_continuous=True)


def is_column_in_explanations(column_name, explanations):
    return any([column_name in explanation for explanation in explanations])


correct_explanations=0

correct_explanations_incorrect_predictions=0
correct_explanations_correct_predictions=0
counter_columns_correct_predictions=0
counter_columns_incorrect_predictions=0

correct_columns=0
counter_correct_predictions=0
counter_incorrect_predictions=0

i=234
#for i in range(len(dataset)):
exp = explainer.explain_instance(X[i], est.predict_proba, num_features=3, top_labels=len(X_features))#num features es tres
#porque es realemente lo que queremos, lo que se sabemos que son importantes.
explanations = [explanation for (explanation, _) in exp.as_list()]


 #prediction de tener un stroke es true (stroke) si su probabilidad es >50%

correct_prediction=True
if (est.predict_proba([X[i]])[0, 1] ) >= 0.5:
    prediction='stroke'
else:
    prediction='healthy'
    
if (prediction=='healthy') and (class_names[y[i]]=='healthy'):
    correct_prediction=True
    counter_correct_predictions=counter_correct_predictions+1
elif (prediction=='stroke') and (class_names[y[i]]=='stroke'):
    counter_correct_predictions=counter_correct_predictions+1
    correct_prediction=True
else:
    correct_prediction=False
    counter_incorrect_predictions=counter_incorrect_predictions+1

#while (correct_prediction!=True):
are_columns_present = []
for column_name in relevant_columns:
        are_columns_present.append(
            is_column_in_explanations(column_name, explanations)
        )
   

# convert to array to permit vectorial operations
are_columns_present = np.array(are_columns_present)
columns_correctly_retrieved = np.sum(are_columns_present)

 #columns correctly retrieved son las veces que estan bmi, average glucose level y age en las explicaciones
correct_columns=correct_columns+columns_correctly_retrieved
if correct_prediction:
    counter_columns_correct_predictions+=columns_correctly_retrieved
    
else:
    counter_columns_incorrect_predictions+=columns_correctly_retrieved
    

if columns_correctly_retrieved == 3:
    correct_explanations=correct_explanations+1
    if correct_prediction:
        correct_explanations_correct_predictions+=1
    else:
        correct_explanations_incorrect_predictions+=1
    

#if not all(are_columns_present):
missing_columns = relevant_columns[np.where(are_columns_present == False)[0]]
print("*************************************************************")
print('Document id: %d' % i)
print('Probability(stroke) =', est.predict_proba([X[i]])[0, 1])
print('True class: %s' % class_names[y[i]])
if (prediction=='healthy') and (class_names[y[i]]=='healthy'):
    print('Prediction is correct')
elif (prediction=='stroke') and (class_names[y[i]]=='stroke'):
    print('Prediction is correct')
else:
    print('Prediction is wrong')
print(f"Example {i}: missing columns are {','.join(missing_columns)}")
print("*************************************************************")

fig = exp.as_pyplot_figure()
#plt.subplots_adjust(left=0.35)
fig.show()
#break
#    
#print('all')
#percentage_total_correct_explanations=correct_explanations/len(dataset)
#percentage_total_correct_columns=correct_columns/(len(dataset)*3)
#print(percentage_total_correct_explanations)   
#print(percentage_total_correct_columns) 
#print('correct prediction')
#print(correct_explanations_correct_predictions/counter_correct_predictions)
#print(counter_columns_correct_predictions/(counter_correct_predictions*3))
#print('incorrect prediction')
#print(correct_explanations_incorrect_predictions/counter_incorrect_predictions)
#print(counter_columns_incorrect_predictions/(counter_incorrect_predictions*3))
#
#
#


##encontrar ejemplos de explicaciones razonables en predicciones incorrectas

