import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


def getDataFromCSV(k):
    f = open("1trainData_"+str(k)+".csv", 'r')
    csvreader = csv.reader(f)
    trainData = list(csvreader)

    f = open("1trainLabels_"+str(k)+".csv", 'r')
    csvreader = csv.reader(f)
    trainLabels = list(csvreader)

    return trainData, trainLabels[0]


for i in [500]:
    Data, Labels = getDataFromCSV(i)

    Data = np.asarray(Data)
    Labels = np.asarray(Labels)

    X_train, X_test, y_train, y_test = train_test_split(
        Data, Labels, test_size=0.33, random_state=42)

    classif = OneVsRestClassifier(SVC(C=2, kernel='rbf', gamma="scale"))

    print("fit...")
    classif.fit(X_train, y_train)

    pre_results = classif.predict(X_test)

    right = 0
    for i in range(len(y_test)):
        if pre_results[i] == y_test[i]:
                right += 1

    print(right/len(y_test)*100)

'''
40.85831863609641
50.91122868900647
58.671369782480895
60.08230452674898
61.140505584950034
62.08112874779541
64.55026455026454
65.02057613168725
65.78483245149911
66.01998824221046
66.78424456202234
68.72427983539094
'''