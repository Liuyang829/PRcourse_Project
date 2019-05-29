import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.optim import lr_scheduler
from torch import nn
import torch
from torch.autograd import Variable
import sys
from gesture_res import *
from torchvision import transforms
from PIL import Image


# net=resnet50(False)
# if torch.cuda.is_available():
#     net=net.cuda() 
# net.eval()


# net.load_state_dict(torch.load("/home/zyh/文档/res/test/model/epoch_276acc92.87581699346406.pth"))
#epoch_264acc92.81045751633987.pth 70%
#epoch_352acc93.05010893246187.pth 70%
#epoch_400acc94.16122004357298.pth 70%
#epoch_348acc94.46623093681917.pth 70%

def softmax(L):
    expL=np.exp(L)
    sumExpL=sum(expL)
    result=[]
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

def predict_image(image_path,net):
    # print("Prediction in progress")
    image = Image.open(image_path)

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

    ])

    # 预处理图像 
    image_tensor = transformation(image).float().cuda()

    # 额外添加一个批次维度，因为PyTorch将所有的图像当做批次
    image_tensor = image_tensor.unsqueeze_(0)
    # print(image_tensor.shape)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # 将输入变为变量
    input = Variable(image_tensor)

    # 预测图像的类
    output = net(input).data.cpu().numpy()
    print(softmax(output.reshape(10)))
    index = output.argmax()

    # print(output)
    return index

def testmodel(model):
    net=resnet101(False)
    if torch.cuda.is_available():
        net=net.cuda() 
    net.eval()

    # net.load_state_dict(torch.load("/home/zyh/文档/res/test/model/"+model))
    net.load_state_dict(torch.load(model))

    err=0
    all=0
    for i in range(1,11):
        path="/home/zyh/文档/res/lytest/lyImg/0"+str(i)+"/"
        if i==10:
            path="/home/zyh/文档/res/lytest/lyImg/"+str(i)+"/"
        for j in os.listdir(path):
            # print(i)
            all+=1
            aaa=int(predict_image(path+j,net))+1
            print(aaa)
            if aaa!=i:
                err+=1
                # print(j,aaa)
    print(model,err/all)

# for model in os.listdir("/home/zyh/文档/res/test/model/"):
#     testmodel(model)
testmodel("/home/zyh/文档/res/test/res101_shou/epoch_148acc93.63557105492589_91.pth")

# for i in range(1,11):
#     path="/home/zyh/文档/res/o/0"+str(i)+"/"
#     if i==10:
#         path="/home/zyh/文档/res/o/"+str(i)+"/"
#     for j in os.listdir(path):
#         # print(i)
#         all+=1
#         aaa=int(predict_image(path+j))+1
#         if aaa!=i:
#             err+=1
#             print(aaa)
# print(err,all)