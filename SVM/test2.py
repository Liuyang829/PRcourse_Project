import numpy as np  # We'll be storing our data as numpy arrays
import os  # For handling directories
from PIL import Image  # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # Plotting
from sklearn.ensemble import RandomForestClassifier
import cv2
from sklearn.externals import joblib
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.externals import joblib


'''
调用存储好的BoW模型和数据集
调整svm参数
进行测试
'''

class BOW(object):
    def __init__(self, ):
        # 创建一个SIFT对象  用于关键点、描述符提取
        self.sift = cv2.xfeatures2d.SIFT_create()

    def getData(self, kmean=10):
        # 创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(kmean)

        for i in range(0, 9):  # Loop over the ten top-level folders
            for j in os.listdir('../0' + str(i) + '/'):
                len = 0
                if not j.startswith('.'):  # Again avoid hidden folders
                    for k in os.listdir('../0' + str(i) + '/' + j + '/'):
                        len += 1
                        path = '../0' + str(i) + '/' + j + '/' + k
                        print(path)
                        bow_kmeans_trainer.add(self.sift_descriptor_extractor(path))
                        if len == 100:
                            break

        # 进行k-means聚类，返回词汇字典 也就是聚类中心
        self.voc = bow_kmeans_trainer.cluster()

        # print( voc.shape)

        # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1, tree=5)
        flann = cv2.FlannBasedMatcher(flann_params, {})
        # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
        self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.sift, flann)
        self.bow_img_descriptor_extractor.setVocabulary(self.voc)
        x_data = []
        y_data = []
        datacount = 0
        # 根据bow_img_descriptor_extractor提取特征向量
        for i in range(0, 9):  # Loop over the ten top-level folders
            for j in os.listdir('../0' + str(i) + '/'):
                if not j.startswith('.'):  # Again avoid hidden folders
                    count = 0  # To tally images of a given gesture
                    for k in os.listdir('../0' + str(i) + '/' + j + '/'):
                        path = '../0' + str(i) + '/' + j + '/' + k
                        descriptor = self.bow_descriptor_extractor(path, kmean)
                        x_data.append(descriptor)
                        count += 1
                    y_values = np.full((count, 1), lookup[j])
                    y_data.append(y_values)
                    datacount += count
        x_data = np.array(x_data, dtype='float32')
        y_data = np.array(y_data).reshape(datacount)
        print(x_data.shape)
        print(y_data.shape)
        return x_data, y_data

    def sift_descriptor_extractor(self, img_path):
        '''
        特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
        '''
        im = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        return self.sift.compute(im, self.sift.detect(im))[1]

    def bow_descriptor_extractor(self, img_path, kmean):
        '''
        提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
        '''
        im = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        return self.bow_img_descriptor_extractor.compute(im, self.sift.detect(im)).reshape(kmean)

    def save(self):
        pickle.dump(self.voc, open('db_yuzhi', 'wb'))

    def load(self):
        with open('db_yuzhi', 'rb') as f:
            voc = pickle.load(f)
            # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
            flann_params = dict(algorithm=1, tree=5)
            flann = cv2.FlannBasedMatcher(flann_params, {})
            # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
            self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.sift, flann)
            self.bow_img_descriptor_extractor.setVocabulary(voc)
            # self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.sift, flann)
            # self.bow_img_descriptor_extractor.setVocabulary(self.voc)


def getLookUp():
    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir('../00/'):
        if not j.startswith('.'):  # If running this code locally, this is to
            # ensure you aren't reading in hidden folders
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1
    return lookup, reverselookup


if __name__ == '__main__':
    lookup, reverselookup = getLookUp()
    print(lookup, reverselookup)
    bow = BOW()
    # bow = pickle.load(open('db', 'rb'))
    bow.load()
    X = pd.read_csv('X_real.csv')
    X = np.array(X)
    Y = pd.read_csv('Y_real.csv')
    Y = np.array(Y)
    Y = np.ravel(Y)
    # clf_rbf = joblib.load('svm_yuzhi.m')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    clf_rbf = svm.SVC(C=5, kernel='rbf', gamma=25, probability=True, decision_function_shape='ovr',
                      class_weight='balanced')
    # clf_liner = svm.SVC(C=1, kernel='linear', gamma=200, decision_function_shape='ovr')
    # clf_rbf=joblib.load('svm_real.m')
    clf_rbf.fit(X_train, y_train)
    # print("clf_rbf: ", clf_rbf.score(X_test, y_test))
    # for i in range(1,11):
            # aaaa = bow.bow_descriptor_extractor(str(i)+".png", 20)
    # joblib.dump(clf_rbf, 'svm_real.m')
    print(clf_rbf.score(X_test,y_test))
    # 照一张图片进行测试判断

    aaaa = bow.bow_descriptor_extractor('../img2/06/gjq_6_16.jpg', 20)
    print(aaaa)
        # print(aaaa)

        # print(bow)
        # joblib.dump(clf_rbf, 'svm_yuzhi.m')
            # pickle.dump(svm,open('svm','wb'))
            # print(aaaa.reshape(1, -1))
    print(clf_rbf.predict(aaaa.reshape(1, -1)))

