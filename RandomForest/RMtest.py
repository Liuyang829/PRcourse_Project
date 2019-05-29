import numpy as np
import pandas as pd
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import csv

# os.getcwd()
# X = pd.read_csv('X_real400.csv')
# X = np.array(X)
# Y = pd.read_csv('Y_real400.csv')
# Y = np.array(Y)
# Y = np.ravel(Y)

def getDataFromCSV():
    f = open("trainData_500.csv", 'r')
    csvreader = csv.reader(f)
    trainData = list(csvreader)

    f = open("trainLabels_500.csv", 'r')
    csvreader = csv.reader(f)
    trainLabels= list(csvreader)

    return trainData,trainLabels[0]

#测试特诊维数与准确率的关系
# max=0
# count1 = 0
# count2 = 0
# for i in range(10,40):
#     for j in range(5,30):
#         print(i," ",j,end=":")
#         rf = RandomForestClassifier(n_estimators=i, max_depth=j)
#         rf.fit(X_train,y_train)
#         if max<rf.score(X_test,y_test):
#             max = rf.score(X_test,y_test)
#             count1 = i
#             count2 = j
#         print(rf.score(X_test,y_test))
#         print('max：',end='')
#         print(max, end='count=')
#         print(count1,count2)

#寻找最佳决策树个数
def resolve1():
    X=[]
    Y=[]
    for i in range(5,200):
        child=[]
        print(i," ",end=":")
        X.append(i)
        rf = RandomForestClassifier(n_estimators=i)
        rf.fit(X_train,y_train)
        Y.append(rf.score(X_test,y_test)*100)
        print(rf.score(X_test,y_test))
    print(X,Y)

    plt.title('RandomForest')
    plt.ylabel('Accuracy Rate%')
    plt.xlabel('Number of Decision Trees')
    plt.plot(X,Y)
    plt.show()

trainData,trainLabels = getDataFromCSV()
X_train, X_test, y_train, y_test = train_test_split(trainData, trainLabels, test_size = 0.25, random_state = 0)
resolve()