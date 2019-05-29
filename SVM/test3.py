import numpy as np
import pandas as pd
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
import csv
from sklearn.multiclass import OneVsRestClassifier

os.getcwd()

'''
调用存储好的数据集
调整svm参数
'''

def getDataFromCSV():
    f = open("trainData_500.csv", 'r')
    csvreader = csv.reader(f)
    trainData = list(csvreader)

    f = open("trainLabels_500.csv", 'r')
    csvreader = csv.reader(f)
    trainLabels= list(csvreader)

    return trainData,trainLabels[0]


X,Y=getDataFromCSV()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)


# gamma = [5, 10, 100, 150,200]
# for x in gamma:
#     for y in range(1,6):
#         #   高斯核
#         clf_rbf = svm.SVC(C=x, kernel='rbf', gamma=y, probability=True, decision_function_shape='ovr',
#                           class_weight='balanced')
#         clf_rbf.fit(X_train, y_train)
#         print("clf_rbf(c=", x, "gamma=", y, "): ", clf_rbf.score(X_test, y_test))
#         #   线性核
#         clf_liner = svm.SVC(C=x, kernel='linear', gamma=y, decision_function_shape='ovr')
#         clf_liner.fit(X_train, y_train)
#         print("clf_liner(c=", x, "gamma=", y, "): ", clf_liner.score(X_test, y_test))
#         #   多项式核
#         clf_poly = svm.SVC(C=x, kernel='poly', degree=y, gamma=100, coef0=0, decision_function_shape='ovr')
#         clf_poly.fit(X_train, y_train)
#         print("clf_poly(c=", x,"degree=",y, "): ", clf_poly.score(X_test, y_test))
#         # sigmoid核
#         clf_sigmoid = svm.SVC(C=x, kernel='sigmoid', gamma=y, probability=True, coef0=0)
#         clf_sigmoid.fit(X_train, y_train)
#         print("clf_sigmoid(c=", x, "gamma=", y, "): ", clf_sigmoid.score(X_test, y_test))
for y in range(1,10):
    clf_poly = OneVsRestClassifier(svm.SVC(C=5, kernel='poly', degree=y,gamma=50))

    clf_poly.fit(X_train, y_train)
    print("clf_poly(degree=",y, "): ", clf_poly.score(X_test, y_test))