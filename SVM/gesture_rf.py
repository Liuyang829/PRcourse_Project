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
import pickle
import random


'''
SIFT提取特征点，放入词袋模型训练出词袋模型并得到特征向量，存入文件中
'''

class BOW(object):
    def __init__(self, ):
        # 创建一个SIFT对象  用于关键点、描述符提取
        self.sift = cv2.xfeatures2d.SIFT_create()

    def getData(self, kmean=10):
        # 创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(kmean)

        # for i in range(0, 10):  # Loop over the ten top-level folders
        #     for j in os.listdir('../o/' + str(i) + '/'):
        #         len = 0
        #         if not j.startswith('.'):  # Again avoid hidden folders
        #             for k in os.listdir('../no/0' + str(i) + '/' + j + '/'):
        #                 len += 1
        #                 path = '../o/0' + str(i) + '/' + j + '/' + k
        #                 print(path)
        #                 bow_kmeans_trainer.add(self.sift_descriptor_extractor(path))
        #                 if len == 10:
        #                     break
        for j in os.listdir('../img2/'):
            len = 0
            if not j.startswith('.'):  # Again avoid hidden folders
                list=os.listdir('../img2/' + j + '/')
                random.shuffle(list)
                for k in list:
                    path = '../img2/' + j + '/' + k
                    len += 1
                    print(path)
                    print(self.sift_descriptor_extractor(path).shape)
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
        # for i in range(0, 10):  # Loop over the ten top-level folders
        #     for j in os.listdir('../new/0' + str(i) + '/'):
        #         if not j.startswith('.'):  # Again avoid hidden folders
        #             count = 0  # To tally images of a given gesture
        #             for k in os.listdir('../new/0' + str(i) + '/' + j + '/'):
        #                 path = '../new/0' + str(i) + '/' + j + '/' + k
        #                 descriptor = self.bow_descriptor_extractor(path, kmean)
        #                 x_data.append(descriptor)
        #                 count += 1
        #             y_values = np.full((count, 1), lookup[j])
        #             y_data.append(y_values)
        #             datacount += count
        for j in os.listdir('../img2/'):
            # len = 0
            if not j.startswith('.'):  # Again avoid hidden folders
                count = 0
                for k in os.listdir('../img2/' + j + '/'):
                    path = '../img2/' + j + '/' + k
                    print(path)
                    descriptor = self.bow_descriptor_extractor(path, kmean)
                    x_data.append(descriptor)
                    count += 1
                y_values = np.full((count, 1), lookup[j])
                y_data.extend(y_values)
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
        print(img_path)
        return self.sift.compute(im, self.sift.detect(im))[1]

    def bow_descriptor_extractor(self, img_path, kmean):
        '''
        提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
        '''
        im = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        return self.bow_img_descriptor_extractor.compute(im, self.sift.detect(im)).reshape(kmean)

    def save(self):
        pickle.dump(self.voc, open('db_real_150', 'wb'))

    def load(self):
        with open('db_real_150', 'rb') as f:
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
    for j in os.listdir('../img2/'):
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
    X, Y = bow.getData(150)
    print(bow)
    bow.save()
    # bow = pickle.load(open('db', 'rb'))
    bow.load()
    print(bow)
    os.getcwd()
    dX = pd.DataFrame(X)
    dY = pd.DataFrame(Y)
    dX.to_csv('X_real_150.csv', index=False, header=True)
    dY.to_csv('Y_real_150.csv', index=False, header=True)

    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    #
    # # for i in range(10, 40):
    # #     for j in range(5, 30):
    # #         print(i, " ", j, end=":")
    # #
    # #         rf = RandomForestClassifier(n_estimators=i, max_depth=j)
    # #         rf.fit(X_train, y_train)
    # #         print(rf.score(X_test, y_test))
    # #   高斯核
    # clf_rbf = svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True, decision_function_shape='ovr')
    # clf_rbf.fit(X_train,y_train)
    # print("clf_rbf: ",clf_rbf.score(X_test,y_test))
    # #   线性核
    # clf_liner = svm.SVC(C=1, kernel='linear', gamma='auto', decision_function_shape='ovr')
    # clf_liner.fit(X_train, y_train)
    # print("clf_liner: ", clf_liner.score(X_test, y_test))
    # #   多项式核
    # clf_poly = svm.SVC(C=1, kernel='poly', degree=3, gamma='auto', coef0=0, decision_function_shape='ovr')
    # clf_poly.fit(X_train, y_train)
    # print("clf_poly: ", clf_poly.score(X_test, y_test))
    # #   sigmoid核
    # clf_sigmoid = svm.SVC(C=0.1, kernel='sigmoid', gamma='scale', probability=True, coef0=0)
    # clf_sigmoid.fit(X_train, y_train)
    # print("clf_sigmoid: ", clf_sigmoid.score(X_test, y_test))
