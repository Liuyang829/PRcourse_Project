# -*- coding: utf-8 -*-
import sys
import os
import pickle
import cv2
import numpy as np
'''
词袋模型BOW+SVM 目标检测

'''

class BOW(object):

    def __init__(self,):
        # 创建一个SIFT对象  用于关键点提取
        self.feature_detector = cv2.xfeatures2d.SIFT_create()
        # 创建一个SIFT对象  用于关键点描述符提取
        self.descriptor_extractor = cv2.xfeatures2d.SIFT_create()

    def fit(self, files, labels, k, length=None):
        '''
            files：训练集图片路径
            labes：对应的每个样本的标签
            k：k-means参数k
            length：指定用于训练词汇字典的样本长度 length<=samples
        '''
        # 类别数
        classes = len(files)

        # 各样本数量
        samples = []
        for i in range(10):
            # samples.append(2)
            samples.append(len(files[i]))
        
        if length is None:
            length = samples        
        # elif  length > samples:
        #     length = samples
        
        # FLANN匹配  参数algorithm用来指定匹配所使用的算法，可以选择的有LinearIndex、KTreeIndex、KMeansIndex、CompositeIndex和AutotuneIndex，这里选择的是KTreeIndex(使用kd树实现最近邻搜索)
        flann_params = dict(algorithm=1,tree=5)
        flann = cv2.FlannBasedMatcher(flann_params,{})
        
        # 创建BOW训练器，指定k-means参数k   把处理好的特征数据全部合并，利用聚类把特征词分为若干类，此若干类的数目由自己设定，每一类相当于一个视觉词汇
        bow_kmeans_trainer = cv2.BOWKMeansTrainer(k)

        print('building BOWKMeansTrainer...')
        # 合并特征数据  每个类从数据集中读取length张图片，通过聚类创建视觉词汇         
        for j in range(classes):        
            for i in range(length):                  
                # 有一些图像会抛异常,主要是因为该图片没有sift描述符
                # print("building BOWKMeansTrainer: ",j+1,i+1,"/",length)
                sys.stdout.write("building BOWKMeansTrainer: "+str(j+1)+":"+str((i+1)/length*100)+"%")
                sys.stdout.write('\r')
                sys.stdout.flush()           
                descriptor = self.sift_descriptor_extractor(files[j][i])                                                          
                if not descriptor is None:
                    bow_kmeans_trainer.add(descriptor)                
                    # print('error:',files[j][i])
        
        # 进行k-means聚类，返回词汇字典 也就是聚类中心
        print("进行k-means聚类...")
        self.voc = bow_kmeans_trainer.cluster()
        
        # 输出词汇字典  <class 'numpy.ndarray'> (40, 128)
        print("输出词汇字典:",type(self.voc),self.voc.shape)
        
        # 初始化bow提取器(设置词汇字典),用于提取每一张图像的BOW特征描述
        self.bow_img_descriptor_extractor = cv2.BOWImgDescriptorExtractor(self.descriptor_extractor,flann)        
        self.bow_img_descriptor_extractor.setVocabulary(self.voc)        
        
        # print('adding features to svm trainer...')
        
        # 创建两个数组，分别对应训练数据和标签，并用BOWImgDescriptorExtractor产生的描述符填充
        # 按照下面的方法生成相应的正负样本图片的标签
        traindata,trainlabels = [],[]
        for j in range(classes):
            for i in range(samples[j]):
                sys.stdout.write("adding features to svm trainer: "+str(j+1)+": "+str((i+1)/samples[j]*100)+"%")
                sys.stdout.write('\r')
                sys.stdout.flush()  

                descriptor = self.bow_descriptor_extractor(files[j][i])
                if not descriptor is None:
                    traindata.extend(descriptor)
                    trainlabels.append(labels[j][i])                

        return  traindata,trainlabels
        
    def sift_descriptor_extractor(self,img_path):
        '''
        特征提取：提取数据集中每幅图像的特征点，然后提取特征描述符，形成特征数据(如：SIFT或者SURF方法)；
        
        args：
            img_path：图像全路径
        '''        
        im = cv2.imread(img_path,0)
        keypoints = self.feature_detector.detect(im)
        # print(type(keypoints))
        if keypoints:
            return self.descriptor_extractor.compute(im,keypoints)[1]
        else:
            return None
    

    def bow_descriptor_extractor(self,img_path):
        '''
        提取图像的BOW特征描述(即利用视觉词袋量化图像特征)
        
        args：
            img_path：图像全路径
        '''        
        im = cv2.imread(img_path,0)
        keypoints = self.feature_detector.detect(im)
        if  keypoints:
            return self.bow_img_descriptor_extractor.compute(im,keypoints)
        else:
            return None

    def save(self,path):
        '''
        保存模型到指定路径
        '''
        print('saving  model....')

        f1 = os.path.join(os.path.dirname(path),path)
        with open(f1,'wb') as f:
            pickle.dump(self.voc,f)

      
            
        
            

        
