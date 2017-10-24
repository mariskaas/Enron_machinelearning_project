# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:17:03 2017

@author: mariska
"""
#%%
#Load the needed librarier and open the data
from time import time
import pandas as pd
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append("../tools/")

with open("C:/Users/mariska/Desktop/Coding_learning/Udacity_machine_learning/ud120-projects/Mariska Final Project Enron_machinelearning_oct17/final_project_dataset_modified_unix.pkl", "rb") as f:
    data_dict = pickle.load(f)
data_df = pd.DataFrame.from_dict(data_dict)
print("Data has been loaded and transformed into a pandas dataframe")

#%% 
#Transform the data to make the persons the index
data_df_transform = data_df.T
print(data_df_transform.columns)
print("data has been transformed")

#%% #Transform all the NaN strings to actual NaN's
data_df_transform.replace(['NaN'], np.nan, inplace=True)
print("NaN strings have been replaced")


#%% #Drop NaN's and test how much data is left
data_df_transform_drop = data_df_transform[['from_poi_to_this_person', 'from_this_person_to_poi', 'poi', 'total_payments', 'total_stock_value']].dropna(how='any')
print("Data before drop:", len(data_df_transform['from_poi_to_this_person']))
print("Data after drop:", len(data_df_transform_drop['from_poi_to_this_person']))

#%% # Make a scatter plot with several columns to see if anything is related and can be used to fill NaN instead of drop
a=data_df_transform['salary']
b=data_df_transform['bonus']
c=data_df_transform['long_term_incentive']
d=data_df_transform['total_payments']
e=data_df_transform['total_stock_value']
plt.scatter(d,c)
plt.show()
plt.scatter(b,d)
plt.show()

#%% Plot shows that salary and long term incentive might be related. Do correlations to see if it is correct.
print(data_df_transform.corr(method='pearson'))

print("Correlations have been done")

#%% #Both salary and bonus are correlated with total payments, may be able to fill NaN with this, first check for a spotted outlier in bonus and total payments
print(data_df_transform[data_df_transform['total_payments']>17000000])
#Outlier is FREVERT MARK A, remove him from the data
data_df_transform.drop('FREVERT MARK A', inplace=True)

#%% #Setting labels and features (changing them to find the best ones)
labels = data_df_transform_drop['poi']

features = data_df_transform_drop[['from_poi_to_this_person', 'from_this_person_to_poi', 'total_stock_value', 'total_payments']]
x=features
y=labels

print("Labels and features have been defined")
#%% Train, test split normal (1split)
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=42)

print("Data has been split with a test size 0.2")
#%% Standard SVM classifier
clf_svm_basic=svm.SVC(kernel='linear', class_weight='balanced')

#Fitting the data, including timing
t0 = time()
clf_svm_basic.fit(x_train,y_train)
print ("Training time:", round(time()-t0, 3), "s")

#Predicting test data, including timing
t0= time()
pred_svm_basic = clf_svm_basic.predict(x_test)
print ("Predicting time:", round(time()-t0, 3), "s")

#Doing the accuracy test
y_pred = pred_svm_basic
y_true = y_test
print("Accuracy score:", metrics.accuracy_score(y_pred, y_true))
print("Precision score:", metrics.precision_score(y_pred, y_true))
print("Recall score:", metrics.recall_score(y_pred, y_true))
print("f1 score:", metrics.f1_score(y_pred, y_true))

#%%Making classifier SVM with folding

clf_svm_fold=svm.SVC(kernel='linear', class_weight="balanced")

#Predictions
scores = cross_val_score(clf_svm_fold, x, y, cv=2, scoring='f1_macro')
#timing the SVM
t0=time()
print ("training time:", round(time()-t0, 3), "s")

#%% #Trying Random Tree Classifier
#making the classififier
clf_random_forest = RandomForestClassifier(class_weight = 'balanced')

#Fitting data and timing 
t0=time()
clf_random_forest.fit(x_train, y_train)
print ("Training time:", round(time()-t0, 3), "s")

#Predicing labels and timing
t0=time()
pred_random_forest = clf_random_forest.predict(x_test)
print ("Predicting time:", round(time()-t0, 3), "s")

y_pred = pred_random_forest
y_true = y_test
print("Accuracy score:", metrics.accuracy_score(y_pred, y_true))
print("Precision score:", metrics.precision_score(y_pred, y_true))
print("Recall score:", metrics.recall_score(y_pred, y_true))
print("f1 score:", metrics.f1_score(y_pred, y_true))

#%% Trying Logistic regression

#making the classifier
clf_log_regression = LogisticRegression(class_weight = 'balanced')

#Fitting data and timing 
t0=time()
clf_log_regression.fit(x_train, y_train)
print ("Training time:", round(time()-t0, 3), "s")

#Predicing labels and timing
t0=time()
pred_log_regression = clf_log_regression.predict(x_test)
print ("Predicting time:", round(time()-t0, 3), "s")

y_pred = pred_log_regression
y_true = y_test
print("Accuracy score:", metrics.accuracy_score(y_pred, y_true))
print("Precision score:", metrics.precision_score(y_pred, y_true))
print("Recall score:", metrics.recall_score(y_pred, y_true))
print("f1 score:", metrics.f1_score(y_pred, y_true))



