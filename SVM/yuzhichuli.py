# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# img = cv2.imread('1.png')
# GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # 中值滤波
# GrayImage= cv2.medianBlur(GrayImage,5)
# ret,th1 = cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
# #3 为Block size, 5为param1值
# th2 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#                     cv2.THRESH_BINARY,3,5)
# th3 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                     cv2.THRESH_BINARY,3,5)
# titles = ['Gray Image', 'Global Thresholding (v = 127)',
# 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [GrayImage, th1, th2, th3]
# for i in range(4):
#    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
# plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


'''
阈值分割预处理（验证效果不佳，已弃用）
'''

# img = cv2.imread('1.png')
# GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # 中值滤波
# GrayImage= cv2.medianBlur(GrayImage,5)
# ret,th1 = cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
# #3 为Block size, 5为param1值
# th2 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#                     cv2.THRESH_BINARY,3,5)
# th3 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                     cv2.THRESH_BINARY,3,5)
# titles = ['Gray Image', 'Global Thresholding (v = 127)',
# 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [GrayImage, th1, th2, th3]
# for i in range(0,4):
#    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#    plt.title(titles[i])
#    plt.xticks([]),plt.yticks([])
# plt.show()
# 自适应阈值对比固定阈值
# for i in range(1,11):
#     img = cv2.imread(str(i)+'.png', 0)

# # 固定阈值
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # 自适应阈值
# th2 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
#     th3 = cv2.adaptiveThreshold(
#         img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)

    # titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i], fontsize=8)
#     plt.xticks([]), plt.yticks([])
# plt.show()

    # cv2.imwrite(str(i)+'.png',th3)


# os.makedirs("../new_img2")
for j in os.listdir('../img2'):
    # print('../new/0' + str(i) + '/'+j)
    # os.makedirs('../new/0' + str(i) + '/'+j)
    if not j.startswith('.'):  # Again avoid hidden folders
        for k in os.listdir('../img2/' + j + '/'):
            # print(k)
            path_old = '../img2/' + j + '/' + k
            img = cv2.imread(path_old, 0)
            th = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
            path_new = '../new_img2/' + j + '/' + k

            cv2.imwrite(path_new,th)
            cv2.imshow(path_new,th)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(path_new)


# for i in range(0, 10):  # Loop over the ten top-level folders
# for j in os.listdir('../o/'):
#     # len = 0
#     if not j.startswith('.'):  # Again avoid hidden folders
#         for k in os.listdir('../o/' + j + '/'):
#             path = '../o/' + j + '/' + k
#             print(path)
            # bow_kmeans_trainer.add(self.sift_descriptor_extractor(path))


# img = cv2.imread('14.png',0)
# GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # 自适应阈值
# th2 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
# th3 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
#
# titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i], fontsize=8)
#     plt.xticks([]), plt.yticks([])
# plt.show()
#
# # th3 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #                     cv2.THRESH_BINARY,3,5)
# cv2.imwrite('svm/14.png',th3)

# # 固定阈值
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # 自适应阈值
# th2 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
# th3 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 6)
#
# titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i], fontsize=8)
#     plt.xticks([]), plt.yticks([])
# plt.show()
#
# cv2.imwrite('15.png',th3)