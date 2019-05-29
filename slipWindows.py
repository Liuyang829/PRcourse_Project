import cv2
from mytest import *
from gesture_res import *
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.optim import lr_scheduler
from torch import nn
import torch
from torch.autograd import Variable
import sys
from gesture_res import *
from torchvision import transforms
from PIL import Image, ImageDraw

def py_nms(dets, thresh, mode="Union"):
    # 非极大值抑制 NMS
    if len(dets) == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]

def slide_window(img, window_size, stride):
    # 构建的金字塔图片滑动窗口。
    window_list = []
    w = img.shape[1]
    h = img.shape[0]
    if w <= window_size + stride or h <= window_size + stride:
        return None
    if len(img.shape) != 3:
        return None
    for i in range(int((w - window_size) / stride)):
        for j in range(int((h - window_size) / stride)):
            box = [j * stride, i * stride, j * stride + window_size, i * stride + window_size]
            window_list.append(box)
    return img, np.asarray(window_list)

def pyramid(image, f, window_size):
    # 构建金字塔模型
    w = image.shape[1]
    h = image.shape[0]
    img_ls = []
    while (w > window_size and h > window_size):
        img_ls.append(image)
        w = int(w * f)
        h = int(h * f)
        image = cv2.resize(image, (w, h))
    return img_ls

def min_gesture(img, F, window_size, stride):
    # 取最小手势
    h, w, d = img.shape
    w_re = int(float(w) * window_size / F)
    h_re = int(float(h) * window_size / F)
    if w_re <= window_size + stride or h_re <= window_size + stride:
        print(None)
    img = cv2.resize(img, (w_re, h_re))
    return img

if __name__ == "__main__":
    # 残差网络模型初始化
    net = resnet101(False)
    if torch.cuda.is_available():
        net=net.cuda() 
    net.eval()

    # 导入残差网络模型
    net.load_state_dict(torch.load("/home/zyh/文档/res/test/res101_shou/epoch_148acc93.63557105492589_91.pth"))

    # 图片路径
    image = cv2.imread('ly_img/07/45.0.jpg')
    h, w, d = image.shape # 长, 宽, 通道数

    # 调参的参数
    # 窗口大小
    IMAGE_SIZE = 20
    # 步长
    stride = 10 
    # 最小手势大小
    F = 100 
    # 图片放缩比例
    ff = 0.7
    # 阈值概率
    p_1 = 0.97
    p_2 = 0.9

    overlapThresh_1 = 0.7
    overlapThresh_2 = 0.3

    # 需要检测的最小
    image_ = min_gesture(image, F, IMAGE_SIZE, stride)

    # 金字塔
    pyd = pyramid(np.array(image_), ff, IMAGE_SIZE)

    # 第一层滑动窗口
    window_after_1 = []
    for i, img in enumerate(pyd):
        # 滑动窗口
        print("layer:",i)
        slide_return = slide_window(img, IMAGE_SIZE, stride)
        if slide_return is None:
            break
        img_1 = slide_return[0]
        window_net_1 = slide_return[1]
        w_1 = img_1.shape[1]
        h_1 = img_1.shape[0]

        patch_net_1 = []
        for box in window_net_1:
            patch = img_1[box[0]:box[2], box[1]:box[3], :]
            patch_net_1.append(patch)
        patch_net_1 = np.array(patch_net_1)
        print(patch_net_1.shape)
        
        window_net = window_net_1
        # 预测手势
        windows = []
        labels = []
        for i, pred in enumerate(patch_net_1):
            # 概率大于阈值的判定为滑动窗口。
            index, possibility = predict_image2(pred, net)
            # print(possibility[index - 1])
            if possibility[index] > p_1:
                # 保存窗口位置和概率。
                windows.append([window_net[i][0], window_net[i][1], window_net[i][2], window_net[i][3], possibility[index]])
                # 保存窗口标签
                labels.append(index)

        # 按照概率值 由大到小排序
        windows = np.asarray(windows)
        windows = py_nms(windows, overlapThresh_1, 'Union')
        window_net = windows
        for box in window_net:
            lt_x = int(float(box[0]) * w / w_1)
            lt_y = int(float(box[1]) * h / h_1)
            rb_x = int(float(box[2]) * w / w_1)
            rb_y = int(float(box[3]) * h / h_1)
            p_box = box[4]
            window_after_1.append([lt_x, lt_y, rb_x, rb_y, p_box])
    # 按照概率值 由大到小排序
    window_net = window_after_1

    # 第二层滑动窗口
    windows_2 = []
    if window_net == []:
        print("Finish")
    if window_net != []:
        patch_net_2 = []
        img_2 = image
        for box in window_net:
            patch = img_2[box[0]:box[2], box[1]:box[3], :]
            patch = cv2.resize(patch, (2, 2))
            patch_net_2.append(patch)
        # 预测手势
        pred_net_2 = []
        for i, pred in enumerate(patch_net_2):
            # 概率大于阈值的判定为滑动窗口。
            index, possibility = predict_image2(pred, net)
            if possibility[index] > p_2:
                # 保存窗口位置和概率。
                windows_2.append([window_net[i][0], window_net[i][1], window_net[i][2], window_net[i][3], possibility[index]])
                # 保存窗口标签
                labels.append(index)
        # 按照概率值 由大到小排序
        windows_2 = np.asarray(windows_2)
        window_net = py_nms(windows_2, overlapThresh_2, 'Union')

    print("window_net:",window_net)
    image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    # 圈出框
    for box in window_net:
        ImageDraw.Draw(image).rectangle((box[1], box[0], box[3], box[2]), outline = "red")
    # 保存图片
    image.save("result.jpg",quality=95)
