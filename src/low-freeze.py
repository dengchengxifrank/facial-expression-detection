import cv2 as cv
import numpy
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import dataset, Dataset
import json
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import Linear
from torch.autograd import Variable
from torch.nn import Softmax
import torch.nn.functional as F



net = models.resnet50()
net.load_state_dict(torch.load('resnet50-0676ba61.pth')) 


# print(net)




class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        # resnet50
        self.net = net
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(1000, 80)
        self.fc2 =nn.Linear(80,7)
        self.dropout = nn.Dropout(0.5)
   

    def forward(self, x):
        x = self.net(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
       
       
   
        return x


model = Model(net)

# model = nn.Sequential(
#     nn.Conv2d(3 ,32, 3),
#     nn.ReLU(),
#     nn.BatchNorm2d(32),
#     nn.Conv2d(32 ,32, 3),
#     nn.ReLU(),
#     nn.BatchNorm2d(32),
#     nn.Conv2d(32 ,32, 5),
#     nn.ReLU(),
#     nn.BatchNorm2d(32),
#     nn.MaxPool2d(kernel_size= (2,2)),
#     nn.Dropout(0.4),
#     nn.Conv2d(32, 64, 3),
#     nn.ReLU(),
#     nn.BatchNorm2d(64),
#     nn.Conv2d(64, 64, 3),
#     nn.ReLU(),
#     nn.BatchNorm2d(64),
#     nn.Conv2d(64, 64, 5),
#     nn.ReLU(),
#     nn.BatchNorm2d(64),
#     nn.MaxPool2d(kernel_size= (2,2),stride=(2,2)),
#     nn.Dropout(0.4),
#     nn.Conv2d(64, 128, 4),
#     nn.ReLU(),
#     nn.BatchNorm2d(128),
#     nn.Flatten(),
#     nn.Linear(282752,256),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(256,7)
# )


for name, value in model.named_parameters():
    print(name)
# 查看有哪些层
# for name, child in model.named_children():
#     print(name)

#fc层不冻结
# for name, child in model.named_children(): 
#     if name in ['fc1,fc2']: 
#         for param in child.parameters():
#             param.requires_grad = True

#网络整体特征是否通用
# for name, child in model.named_children(): 
#     if name in ['fc1,fc2']: 
#         for param in child.parameters():
#             param.requires_grad = True
#     else:
#         for param in child.parameters():
#             param.requires_grad = False

#高层不冻结           
# for name, child in model.net.named_children():
#     if name in ['layer4','fc']:
#         for param in child.parameters():
#             param.requires_grad = True
#     else:
#         for param in child.parameters():
#             param.requires_grad = False

#底层不冻结
for name, child in model.net.named_children():
    if name in ['layer1','conv1','bn1']:
        for param in child.parameters():
            param.requires_grad = True
    else:
        for param in child.parameters():
            param.requires_grad = False   
#中层不冻结         
# for name, child in model.net.named_children():
#     if name in ['layer2','layer3']:
#         for param in child.parameters():
#             param.requires_grad = True
#     else:
#         for param in child.parameters():
#             param.requires_grad = False       
# print("模型冻结完成")

# #查看哪些层被冻结
# for name, value in model.named_parameters():
#     print(name, "\t冻结=\t",value.requires_grad)

labels = []

tf = open('./train.json', 'r')

s = json.load(tf)

for i in s.values():
    labels.append(i)


labels_test = []
tt = open('./test.json','r')
tests = json.load(tt)
for i in tests.values():
    labels_test.append(i)


train_acc = []

test_acc = []




class testset(Dataset):
    def __init__(self,transform):
        self.image = os.listdir('./testset')
        self.labels_test = labels_test
        self.images_labels = []
        self.transform = transform
        for i in range(len(self.labels_test)):
            self.images_labels.append(['./testset/' + self.image[i], labels_test[i]])
        
    def __getitem__(self, index):
        image_path, labels_test = self.images_labels[index][0], self.images_labels[index][1]
      
      

        images = Image.open(image_path).convert('RGB')

        images = self.transform(images)

        labels_test = torch.Tensor(labels_test)

        return images, labels_test

    def __len__(self):
        return len(self.labels_test)



class trainset(Dataset):
    def __init__(self,transform):
        self.image = os.listdir('./trainset')
        self.labels = labels
        self.transform = transform
        self.images_labels = []
        for i in range(len(self.labels)):
            self.images_labels.append(['./trainset/' + self.image[i], labels[i]])

    def __getitem__(self, index):
        image_path, labels = self.images_labels[index][0], self.images_labels[index][1]

        images = Image.open(image_path).convert('RGB')

      
        images = self.transform(images)
        
    

        labels = torch.Tensor(labels)
        return images, labels

    def __len__(self):
        return len(self.labels)

        
train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), # 将图像随意裁剪，宽高均为299
                    transforms.RandomHorizontalFlip(), # 以 0.5 的概率左右翻转图像
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                    transforms.RandomRotation(degrees=5, expand=False, fill=None),
                    transforms.ToTensor(), # 将 PIL 图像转为 Tensor，并且进行归一化
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
                ])
test_transform = transforms.Compose([
                    transforms.Resize(256), 
                    transforms.CenterCrop(224),
                    transforms.ToTensor(), # 将 PIL 图像转为 Tensor，并且进行归一化
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
                ])

train_data = trainset(transform=train_transform)

train_loader = DataLoader(train_data, batch_size=2)


test_data = testset(transform=test_transform)

test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=True)


# Learning_rate = 0.0001

optimizer = optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-4)
#网络整体特征是否通用
# optimizer = optim.Adam([                         
                        
#                             {'params': model.fc1.parameters(),'lr': 1e-4},
#                             {'params': model.fc2.parameters(),'lr': 1e-4},
                     
#                         ],  weight_decay=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer,2,gamma=0.94)


criterion = nn.MultiLabelSoftMarginLoss()

#高层不冻结优化器
# optimizer = torch.optim.Adam([                         
#                             {'params': model.net.fc.parameters(),'lr': 1e-4},
#                             {'params': model.fc1.parameters(),'lr': 1e-4},
#                             {'params': model.fc2.parameters(),'lr': 1e-4},
#                             {'params': model.net.layer4.parameters(),'lr': 1e-5},
#                         ],  weight_decay=1e-4)
#低层不冻结优化器
optimizer = torch.optim.Adam([ {'params': model.net.fc.parameters(),'lr': 1e-4},                         
                            {'params': model.net.conv1.parameters(),'lr': 1e-4},
                            {'params': model.net.bn1.parameters(),'lr': 1e-4},
                            {'params': model.layer1.parameters(),'lr': 1e-5},
                        
                        ],weight_decay=1e-4)
# 中层不冻结优化器
# optimizer = torch.optim.Adam([ {'params': model.net.fc.parameters(),'lr': 1e-4},                         
#                             {'params': model.net.layer2.parameters(),'lr': 1e-5},
#                             {'params': model.net.layer3.parameters(),'lr': 1e-5},
#                            
                        
#                         ], weight_decay=1e-4)



def draw_trainloss(x, y):
    plt.clf()
    plt.switch_backend('agg')
    plt.plot(x, y)
    plt.savefig("./train_loss.png")
    plt.ioff()


def draw_trainacc(x, y):
    plt.clf()
    plt.switch_backend('agg')
    plt.plot(x, y)
    plt.savefig("./train_acc.png")
    plt.ioff()


def draw_testloss(x, y):
    plt.clf()
    plt.switch_backend('agg')
    plt.plot(x, y)
    plt.savefig("./test_loss.png")
    plt.ioff()

def drwa_testacc(x, y):
    plt.clf()
    plt.switch_backend('agg')
    plt.plot(x, y)
    plt.savefig("./test_acc.png")
    plt.ioff()


x=[]

y_trainloss=[]

y_trainacc=[]

y_testloss=[]

y_testacc=[]






for epoch in range(150):
   
    run_loss = 0
    train_correct = 0
    model.train()
    for i, data in enumerate(train_loader):
        images, labels = data[0], data[1]
        labels = torch.Tensor(labels)
        images, labels = Variable(images.float()), Variable(labels.float())

        # images, labels = images.cuda(), labels.cuda()
     
        outputs = model(images)

        outputs = outputs.float()
        # print(outputs.shape)
        # print(labels.shape)
        
        
        # outputs = outputs[0]
        
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        
        
        
        with torch.no_grad():
            outputs_pre = torch.argmax(outputs, dim=1)
            labels = torch.argmax(labels,dim=1)
            #print(outputs,outputs_pre,torch.argmax(labels))
            train_correct += (outputs_pre == labels).sum().item()
            
            #run_loss += loss.item()
    
    epoch_loss = run_loss / len(train_loader.dataset)
    epoch_acc = train_correct /len(train_loader.dataset)
    print('this is from epoch %d'%(epoch))
    print('train epoch_loss is',epoch_loss,'train epoch_acc is ',epoch_acc)
    
    x.append(epoch)
    y_trainloss.append(epoch_loss)
    y_trainacc.append(epoch_acc)
    draw_trainloss(x,y_trainloss)
    
    draw_trainacc(x,y_trainacc)
    
    
    test_correct = 0

    
    test_running_loss = 0

    
    #接下来进行test
   
 
    with torch.no_grad():
        model.eval()
        cnt = 0
        for i, data in enumerate(test_loader):
                            
            images, labels = data[0], data[1]
                                    
                        
            outputs_pre = model(images)
                        
            outputs_pre = outputs_pre.float()
            
            loss_tmp = criterion(outputs_pre,labels)            



            outputs_pre = torch.argmax(outputs_pre, dim=1)
            labels = torch.argmax(labels,dim=1)
            test_correct += (outputs_pre == labels).sum().item()
            test_running_loss += loss.item()

    epoch_test_loss = test_running_loss / len(test_loader.dataset)
    epoch_test_acc = test_correct / len(test_loader.dataset)
    
    y_testloss.append(epoch_test_loss)
    
    y_testacc.append(epoch_test_acc)
    
    draw_testloss(x,y_testloss)
    drwa_testacc(x,y_testacc)
    
    print('test loss is ',epoch_test_loss,'  test acc is ',epoch_test_acc)
    
    
    print('                           ')
    torch.save(net,'./model/best.pt')