# -*- coding: utf-8 -*-
from pyramid import pyramid, silding_window
import os
from non_max_suppression import nms
from detector import BOW
import numpy as np
import cv2
import csv


def prepare_GestData(path):
    print("载入数据集GestData...")
    files = []
    labels = []
    length = []
    for i in range(1, 11):
        fileslist = []
        labelslist = []
        for file in os.listdir(path+str(i)+"/"):
            fileslist.append(path+str(i)+"/"+file)
            labelslist.append(i)
        files.append(fileslist)
        labels.append(labelslist)
        length.append(len(fileslist))

    print("载入数据集完成：共10类。每类样本数为：", length, "sum:", sum(length))

    return files, labels, length


if __name__ == '__main__':

    bow = BOW()
    files = []
    labels = []

    train_Percent = 1000

    rootpath = 'GestData/'

    filesAll, labelsAll, length = prepare_GestData(rootpath)

    for kk in [500]:
        print("\n构建bow, k=", kk, "...")
        trainData, trainLabels = bow.fit(filesAll, labelsAll, kk, 200)

        bow.save("1dict_"+str(kk)+".pkl")
        print("length of trainData:", len(trainData), len(trainData[0]))
        print("length of trainLabels:", len(trainLabels))

        with open('1trainData_' + str(kk) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in trainData:
                writer.writerow(row)

        with open('1trainLabels_' + str(kk) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(trainLabels)
