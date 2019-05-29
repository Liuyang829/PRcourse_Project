import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms
from torch.optim import lr_scheduler
from torch import nn
import torch
from torch.autograd import Variable
import sys
from gesture_res import *
import os # For handling directories

EPOCH = 500   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 50     #批处理尺寸(batch_size)
LR = 0.001        #学习率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
}

data_dir = '/home/zyh/文档/res/train/img2'
image_datasets = datasets.ImageFolder(data_dir,data_transforms["train"])

test_dir = "/home/zyh/文档/res/lytest/lyImg/"
test_datasets = datasets.ImageFolder(test_dir,data_transforms["test"])


dataloders =  torch.utils.data.DataLoader(image_datasets,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=20)

testloders =   torch.utils.data.DataLoader(test_datasets,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=20)
                                            

print(image_datasets.class_to_idx)
net=resnet101(False)
if torch.cuda.is_available():
    net=net.cuda() 

# net.load_state_dict(torch.load("/home/zyh/文档/res/test/model/epoch_44acc86.135.pth"))
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
# optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

optimizer=torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
# scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

with open("acc.txt","w") as f:
    with open("test.txt","w") as f2:
        for epoch in range(EPOCH):
            print('\nEpoch: %d' % (epoch + 1))
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            # scheduler.step()
            for i, data in enumerate(dataloders):
                length = len(dataloders)
                inputs, labels=data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
            accuracy=100. * int(correct) / total
            print('train:[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), accuracy))
            f.write('[ epoch: %d , iter: %d ] Loss: %.03f | Acc: %.3f '
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), accuracy))
            f.write('\n')
            f.flush()
            with torch.no_grad(): 
                correct = 0 
                total = 0 
                for data in testloders: 
                    net.eval() 
                    images, labels = data 
                    images, labels = images.to(device), labels.to(device) 
                    outputs = net(images) # 取得分最高的那个类 (outputs.data的索引号) 
                    _, predicted = torch.max(outputs.data, 1) 
                    total += labels.size(0) 
                    correct += (predicted == labels).sum() 
                    acc = 100. * correct / total # 将每次测试结果实时写入acc.txt文件中 
                print('测试分类准确率为：%.3f%%' % (100 * correct/total)) 
                f2.write("EPOCH= %d ,Accuracy= %.3f " % (epoch + 1, acc)) 
                f2.write('\n') 
                f2.flush()


            
            
            
            
            
        

