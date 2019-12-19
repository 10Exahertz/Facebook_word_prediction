#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:40:50 2019

@author: stevenalsheimer
"""
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.datasets import make_classification
import csv
Z = []
v = []
with open('megadoc_train_U23.txt') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        v.append(row[0])
        row.remove(row[0])
        test_list = [int(i) for i in row]
        Z.append(test_list)
X = np.array(Z)
y = np.array(v)
print("past stage 1")
clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,tol=1e-3)
clf.fit(X, y)
PassiveAggressiveClassifier(random_state=0)
print(clf.coef_)
#[[0.26642044 0.45070924 0.67251877 0.64185414]]
print(clf.intercept_)
#[1.84127814]
print("Past Stage 2")

with open('Megadoc_NLP_create_U23.txt') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    total = 0
    correct = 0
    for row in readCSV:
        total += 1
        class_name = row[0]
        row.remove(row[0])
        test_list = [int(i) for i in row]
        #print(total, class_name, [test_list])
        pred_name = clf.predict([test_list])[0]
        if class_name == pred_name:
            correct += 1
    score  = correct/total
print(score)
        
#print(clf.predict([[1, 0, 0, 1, 1, 0, 1]])[0])
#[1]
