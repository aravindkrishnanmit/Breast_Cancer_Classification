#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:51:20 2018

@author: yash
"""

#Logistic Regression

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('breastCancer.csv')
dataset.replace('?', 0, inplace=True)
dataset = dataset.applymap(np.int64)#converts all double data type to integer type data
X = dataset.iloc[:, 1:-1].values    
y = dataset.iloc[:, -1].values
#handling labels column
y_new = []
for i in range(len(y)):
    if y[i] == 2:
        y_new.append(0)
    else:
        y_new.append(1)
y_new = np.array(y_new)


#Splitting the dataset into the Training Set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y_new, test_size=0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting Logistic Regression To The Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)

#Predicting The Test Set Results
y_pred = classifier.predict(X_test)

#Making The Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

#Evaluation Matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy: {:.2f}'.format(accuracy_score(Y_test,y_pred)))
print('Prediction: {:.2f}'.format(precision_score(Y_test,y_pred)))
print('Recall: {:.2f}'.format(recall_score(Y_test,y_pred)))
print('F1: {:.2f}'.format(f1_score(Y_test,y_pred)))