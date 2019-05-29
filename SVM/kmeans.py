from sklearn.cluster import KMeans
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
import matplotlib.pyplot as plt


'''
使用SEE“手肘模型”分析K-means中K的取值
'''


class BOW(object):
    def __init__(self, ):
        # 创建一个SIFT对象  用于关键点、描述符提取
        self.sift = cv2.xfeatures2d.SIFT_create()

    def sift_descriptor_extractor(self, img_path):
        '''
        特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
        '''
        im = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        print(img_path)
        return self.sift.compute(im, self.sift.detect(im))[1]

    def Kmeans_test(self):
        inertia = []
        label_pred = []
        meanall = []
        centroids = []
        for j in os.listdir('../img2/'):
            len = 0
            if not j.startswith('.'):  # Again avoid hidden folders
                list = os.listdir('../img2/' + j + '/')
                random.shuffle(list)
                for k in list:
                    path = '../img2/' + j + '/' + k
                    len += 1
                    print(path)
                    # print(meanall)
                    # print(self.sift_descriptor_extractor(path))
                    meanall.extend(self.sift_descriptor_extractor(path))
                    if len == 30:
                        break
        meanall = np.array(meanall)
        print(type(meanall),meanall)
        list=[20,50,100,150,200,300,400,500,600,1000,1500,2000]
        for k in list:
            print(k)
            estimator = KMeans(n_clusters=k)  # 构造聚类器
            estimator.fit(meanall)  # 聚类
            label_pred.append(estimator.labels_)  # 获取聚类标签
            centroids.append(estimator.cluster_centers_)  # 获取聚类中心
            inertia.append(estimator.inertia_)  # 获取聚类准则的总和
            print(estimator.inertia_)

        print(meanall)
        print(inertia)
        plt.plot(list,inertia)
        plt.show()

#
bow = BOW()
bow.Kmeans_test()
# inertia3_6=[2807440981.3381715, 2796506538.2439876, 2780945733.6617894, 2767701851.3814955, 2756937911.376001, 2744607911.439963, 2732137720.1895113, 2719606419.855219, 2711038134.6826506, 2702043412.1143656, 2693115787.155238, 2681774048.351797, 2674119151.492858, 2662191030.5036306, 2651660307.3031583, 2644244066.4507556, 2639959319.1411686, 2626946534.9398866, 2622493752.9224496, 2609989567.435099, 2606021879.907249, 2597840595.3112087, 2590655619.825697, 2584255486.1984296, 2578685428.081598, 2571439774.33185, 2563402479.9795794, 2562275181.183121, 2555074101.112423, 2548234910.6256666]
# inertia1_2=[4788507000.0, 4318260475.117521, 3961525545.53348, 3797382455.6868644, 3660433407.0924478, 3554802038.714291, 3473910572.145875, 3400778969.5176764, 3334299786.793602, 3269563603.38221, 3206041672.4388824, 3152522364.9192305, 3110867415.5096865, 3071513666.350345, 3038016247.603059, 3002667862.96981, 2975213009.078604, 2949459153.803087, 2926187809.815773]
# x=range(10,300,10)
#
# inertia=[3181334615.7909837, 2835016827.2197914, 2662780609.9666114, 2546392285.332779, 2468323737.532231, 2410843673.5367236, 2357523903.8650603, 2315970989.7643023, 2279071939.4661336, 2246701981.8490033, 2217988072.271117, 2191088909.0531445, 2169469012.1779327, 2147864943.5073023, 2128678996.6706283, 2109305382.4843593, 2090782868.0389545, 2075457014.3925345, 2060492311.933862, 2046981132.4789784, 2031514730.182295, 2020471172.3441532, 2007457383.683257, 1995266104.0765564, 1982893791.0792518, 1973144994.5107062, 1962793472.0599818, 1951410733.8594804, 1943011185.2518618]
# plt.plot(x,inertia)
# plt.show()


# class BOW(object):
#     def __init__(self, ):
#         # 创建一个SIFT对象  用于关键点、描述符提取
#         self.sift = cv2.xfeatures2d.SIFT_create()
#
#     def getData(self, kmean=10):
#         # 创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
#         bow_kmeans_trainer = cv2.BOWKMeansTrainer(kmean)
#
#         # for i in range(0, 10):  # Loop over the ten top-level folders
#         #     for j in os.listdir('../o/' + str(i) + '/'):
#         #         len = 0
#         #         if not j.startswith('.'):  # Again avoid hidden folders
#         #             for k in os.listdir('../no/0' + str(i) + '/' + j + '/'):
#         #                 len += 1
#         #                 path = '../o/0' + str(i) + '/' + j + '/' + k
#         #                 print(path)
#         #                 bow_kmeans_trainer.add(self.sift_descriptor_extractor(path))
#         #                 if len == 10:
#         #                     break
#         for j in os.listdir('../img2/'):
#             len = 0
#             if not j.startswith('.'):  # Again avoid hidden folders
#                 list = os.listdir('../img2/' + j + '/')
#                 random.shuffle(list)
#                 for k in list:
#                     path = '../img2/' + j + '/' + k
#                     len += 1
#                     print(path)
#                     bow_kmeans_trainer.add(self.sift_descriptor_extractor(path))
#                     if len == 100:
#                         break
#
#         # 进行k-means聚类，返回词汇字典 也就是聚类中心
#         self.voc = bow_kmeans_trainer.cluster()
#
#         # print( voc.shape)
#
#         # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
#         flann_params = dict(algorithm=1, tree=5)
#         flann = cv2.FlannBasedMatcher(flann_params, {})
#         # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
#         self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.sift, flann)
#         self.bow_img_descriptor_extractor.setVocabulary(self.voc)
#         x_data = []
#         y_data = []
#         datacount = 0
#         # 根据bow_img_descriptor_extractor提取特征向量
#         # for i in range(0, 10):  # Loop over the ten top-level folders
#         #     for j in os.listdir('../new/0' + str(i) + '/'):
#         #         if not j.startswith('.'):  # Again avoid hidden folders
#         #             count = 0  # To tally images of a given gesture
#         #             for k in os.listdir('../new/0' + str(i) + '/' + j + '/'):
#         #                 path = '../new/0' + str(i) + '/' + j + '/' + k
#         #                 descriptor = self.bow_descriptor_extractor(path, kmean)
#         #                 x_data.append(descriptor)
#         #                 count += 1
#         #             y_values = np.full((count, 1), lookup[j])
#         #             y_data.append(y_values)
#         #             datacount += count
#         for j in os.listdir('../img2/'):
#             # len = 0
#             if not j.startswith('.'):  # Again avoid hidden folders
#                 count = 0
#                 for k in os.listdir('../img2/' + j + '/'):
#                     path = '../img2/' + j + '/' + k
#                     print(path)
#                     descriptor = self.bow_descriptor_extractor(path, kmean)
#                     x_data.append(descriptor)
#                     count += 1
#                 y_values = np.full((count, 1), lookup[j])
#                 y_data.extend(y_values)
#                 datacount += count
#         x_data = np.array(x_data, dtype='float32')
#         y_data = np.array(y_data).reshape(datacount)
#         print(x_data.shape)
#         print(y_data.shape)
#         return x_data, y_data
#
#     def sift_descriptor_extractor(self, img_path):
#         '''
#         特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
#         '''
#         im = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
#         print(img_path)
#         return self.sift.compute(im, self.sift.detect(im))[1]
#
#     def bow_descriptor_extractor(self, img_path, kmean):
#         '''
#         提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
#         '''
#         im = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
#         return self.bow_img_descriptor_extractor.compute(im, self.sift.detect(im)).reshape(kmean)
#
#     def save(self):
#         pickle.dump(self.voc, open('db_real_30', 'wb'))
#
#     def load(self):
#         with open('db_real_30', 'rb') as f:
#             voc = pickle.load(f)
#             # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
#             flann_params = dict(algorithm=1, tree=5)
#             flann = cv2.FlannBasedMatcher(flann_params, {})
#             # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
#             self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.sift, flann)
#             self.bow_img_descriptor_extractor.setVocabulary(voc)
#             # self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.sift, flann)
#             # self.bow_img_descriptor_extractor.setVocabulary(self.voc)
#
#     def getTest(self):
#         inertia = []
#         label_pred = []
#         meanall = []
#         centroids = []
#         label = -1
#         y = []
#         for j in os.listdir('../lyImg/'):
#             len = 0
#             label += 1
#             if not j.startswith('.'):  # Again avoid hidden folders
#                 list = os.listdir('../lyImg/' + j + '/')
#                 random.shuffle(list)
#                 for k in list:
#                     path = '../lyImg/' + j + '/' + k
#                     len += 1
#                     print(path)
#                     # print(meanall)
#                     # print(self.sift_descriptor_extractor(path))
#                     meanall.append(np.array(self.bow_descriptor_extractor(path,30)))
#                     y.extend(np.array([label]))
#                     if len == 30:
#                         break
#         meanall = np.array(meanall)
#         y = np.array(y)
#         print(meanall.shape, meanall)
#         print(y.shape, y)
#         return meanall, y
#
#
# def getLookUp():
#     lookup = dict()
#     reverselookup = dict()
#     count = 0
#     for j in os.listdir('../img2/'):
#         if not j.startswith('.'):  # If running this code locally, this is to
#             # ensure you aren't reading in hidden folders
#             lookup[j] = count
#             reverselookup[count] = j
#             count = count + 1
#     return lookup, reverselookup
#
#
# lookup, reverselookup = getLookUp()
# print(lookup, reverselookup)
# bow = BOW()
# bow.load()
# print(bow)
# X_test, y_test = bow.getTest()
# os.getcwd()
# dX = pd.DataFrame(X_test)
# dY = pd.DataFrame(y_test)
# dX.to_csv('X_test_30.csv', index=False, header=True)
# dY.to_csv('Y_test_30.csv', index=False, header=True)
