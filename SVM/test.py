import numpy as np
import pandas as pd
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
import csv
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

'''
调用存储好的数据集
调整svm参数
'''

os.getcwd()
# t = np.array([[1, 2, 3], [4, 5, 6]])
# dt=pd.DataFrame(t)
# dt.to_csv('result.csv',index=False,header=True)
# print(t)
#
# X = pd.read_csv('X_real_30.csv')
# X = np.array(X)
# Y = pd.read_csv('Y_real_30.csv')
# Y = np.array(Y)
# Y = np.ravel(Y)
# X_train=X
# y_train=Y
# # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
# XX = pd.read_csv('X_test_30.csv')
# XX = np.array(XX)
# YY = pd.read_csv('Y_test_30.csv')
# YY = np.array(YY)
# YY = np.ravel(YY)
# X_test=XX
# y_test=YY random_state=0,

def getDataFromCSV():
    f = open("trainData_500.csv", 'r')
    csvreader = csv.reader(f)
    trainData = list(csvreader)

    f = open("trainLabels_500.csv", 'r')
    csvreader = csv.reader(f)
    trainLabels= list(csvreader)

    return trainData,trainLabels[0]


X,Y=getDataFromCSV()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=40)
# dX = pd.DataFrame(X_train)
# dY = pd.DataFrame(y_train)
# dX.to_csv('X_real_train_300.csv', index=False, header=True)
# dY.to_csv('Y_real_train_300.csv', index=False, header=True)
#
# dXX = pd.DataFrame(X_test)
# dYY = pd.DataFrame(y_test)
# dXX.to_csv('X_real_test_300.csv', index=False, header=True)
# dYY.to_csv('Y_real_test_300.csv', index=False, header=True)
#
#
# X_train = pd.read_csv('X_real_train_300.csv')
# X_train = np.array(X_train)
# y_train = pd.read_csv('Y_real_train_300.csv')
# y_train = np.array(y_train)
# y_train = np.ravel(y_train)
#
# X_test = pd.read_csv('X_real_test_300.csv')
# X_test = np.array(X_test)
# y_test = pd.read_csv('Y_real_test_300.csv')
# y_test = np.array(y_test)
# y_test = np.ravel(y_test)

# clf_rbf = OneVsRestClassifier(svm.SVC(C=5,  kernel='rbf', gamma=35))
# clf_rbf=svm.SVC(C=5,kernel='rbf',gamma=35,probability=True,decision_function_shape='ovr')
# clf_rbf.fit(X_train, y_train)
# print("clf_rbf(c=", 2, "gamma=scale): ", clf_rbf.score(X_test, y_test))
# result=clf_rbf.predict(X_test)
# acc=0
# print(result)
# print(y_test)
# for i in range(len(result)):
#     if(result[i]==y_test[i]):
#         acc+=1
# proba=clf_rbf.predict_proba(X_test)
# print(acc/len(result))
# print("clf_rbf(c=", 2, "gamma=scale): ", clf_rbf.score(X_test, y_test))
# print(proba)

# draw_roc(clf_rbf,y_test,proba)

gamma = [5, 10, 15, 20,100]
for x in range(10,15):
    for y in range(4,6):
        # #   高斯核
        # clf_rbf =  OneVsRestClassifier(svm.SVC(C=x,  kernel='rbf', gamma=y))
        # clf_rbf.fit(X_train, y_train)
        # print("clf_rbf(c=", x, "gamma=", y, "): ", clf_rbf.score(X_test, y_test))
        # #   线性核
        # clf_liner = OneVsRestClassifier(svm.SVC(C=x, kernel='linear', gamma=y))
        # clf_liner.fit(X_train, y_train)
        # print("clf_liner(c=", x, "gamma=", y, "): ", clf_liner.score(X_test, y_test))
        # #   多项式核
        # clf_poly = OneVsRestClassifier(svm.SVC(C=x, kernel='poly', degree=6, gamma=100))
        # clf_poly.fit(X_train, y_train)
        # print("clf_poly(c=", x, "): ", clf_poly.score(X_test, y_test))
        # sigmoid核
        clf_sigmoid = OneVsRestClassifier(svm.SVC(C=x, kernel='sigmoid', gamma=y))
        clf_sigmoid.fit(X_train, y_train)
        print("clf_sigmoid(c=", x, "gamma=", y, "): ", clf_sigmoid.score(X_test, y_test))
