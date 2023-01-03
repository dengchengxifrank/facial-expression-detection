from re import T
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


#多卡

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

torch.backends.cudnn.enabled = False

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose({
    transforms.Resize((224, 224)),  # Resize
    #transforms.RandomCrop(32, padding=4),  # RandomCrop，对数据进行随机的裁剪
    #transforms.ToPILImage,
    transforms.ToTensor(),  # ToTensor，将图片转成张量的形式同时会进行归一化操作，把像素值的区间从0-255归一化到0-1
    transforms.Normalize(norm_mean, norm_std),  # 标准化操作，将数据的均值变为0，标准差变为1
})  # Resize的功能是缩放，RandomCrop的功能是裁剪，ToTensor的功能是把图片变为张量

# netword
#net1 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)




net = models.vgg16(pretrained=None)

net1 = models.vgg16(pretrained=True)

net.classifier.add_module = nn.Sequential(
    nn.Linear(1000,80),
    nn.LeakyReLU(),
    nn.Dropout(),
    nn.Linear(80,7)
)


new_dict = net1.state_dict()

tmp = net.state_dict()

for k in new_dict.keys():
     if k in tmp.keys():
         tmp[k] = new_dict[k]
#这里net就是最后的网络
net.load_state_dict(tmp)

#net = torch.nn.DataParallel(net)

#net = net.cuda()

print(net)


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

seed = 10
torch.manual_seed(seed)


class testset(Dataset):
    def __init__(self):
        self.image = os.listdir('../trainset')
        self.labels_test = labels_test
        self.images_labels = []
        for i in range(len(self.labels_test)):
            self.images_labels.append(['../trainset/' + self.image[i], labels_test[i]])
        
    def __getitem__(self, index):
        image_path, labels_test = self.images_labels[index][0], self.images_labels[index][1]
        images = cv.imread(image_path)

        images = cv.resize(images, (224, 224))
        images = images / 255

        images = np.transpose(images, (2, 0, 1))

        images = torch.Tensor(images)

        labels_test = torch.Tensor(labels_test)

        return images, labels_test

    def __len__(self):
        return len(self.labels_test)


testset = testset()

testloader = DataLoader(dataset=testset, batch_size=4, shuffle=False, num_workers=0)

class trainset(Dataset):
    def __init__(self):
        self.image = os.listdir('../trainset')
        self.labels = labels
        self.images_labels = []
        for i in range(len(self.labels)):
            self.images_labels.append(['../trainset/' + self.image[i], labels[i]])

    def __getitem__(self, index):
        image_path, labels = self.images_labels[index][0], self.images_labels[index][1]

        images = cv.imread(image_path)

        #images = train_transform(images)
        #images = cv.cvtColor(images, cv.COLOR_BGR2RGB)
        images = cv.resize(images, (224, 224))
        images = images/255
        images = np.transpose(images, (2, 0, 1))

        images = torch.Tensor(images)

        labels = torch.Tensor(labels)
        return images, labels

    def __len__(self):
        return len(self.labels)


s = trainset()


train_loader = DataLoader(dataset=s, batch_size=8, shuffle=True, num_workers=0, drop_last=False)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss()

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



lossmax = 10000


for epoch in range(100):
    
    run_loss = 0
    train_correct = 0
    net.train()
    for i, data in enumerate(train_loader):
        images, labels = data[0], data[1]
        labels = torch.Tensor(labels)
        images, labels = Variable(images.float()), Variable(labels.float())

        #images, labels = images.cuda(), labels.cuda()
                        
        outputs = net(images)
                
        loss = criterion(outputs,labels)
         
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        
        
        with torch.no_grad():

            outputs_pre = torch.argmax(outputs,dim=1)
            labels = torch.argmax(labels,dim=1)

            train_correct += (outputs_pre == labels).sum().item()
            
            
            run_loss += loss.item()
    
    epoch_loss = run_loss / len(train_loader.dataset)
    epoch_acc = train_correct / len(train_loader.dataset)
    print('train epoch_loss is',epoch_loss,'train epoch_acc is ',epoch_acc)
    
    x.append(epoch)
    y_trainloss.append(epoch_loss)
    y_trainacc.append(epoch_acc)
    draw_trainloss(x,y_trainloss)
    draw_trainacc(x,y_trainacc)
    
    
    test_correct = 0
    test_total = 23
    
    test_running_loss = 0

    
    #接下来进行test
   
 
    with torch.no_grad():
        net.eval()
        cnt = 0
        for i, data in enumerate(testloader):
                            
            images, labels = data[0], data[1]
                                    
            images,labels = images.detach(),labels.detach()
                        
            outputs_pre = net(images)
                        
            outputs_pre = outputs_pre.float()
            
            loss_tmp = criterion(outputs_pre,labels)            

            outputs_pre = torch.argmax(outputs_pre, dim=1)
            labels = torch.argmax(labels,dim=1)
           #print(outputs_pre,'           ',labels)
            test_correct += (outputs_pre == labels).sum().item()
            test_running_loss += loss.item()
    print('in epoch %d correct itemnum  '%(test_correct))
    epoch_test_loss = test_running_loss / len(testloader.dataset)
    epoch_test_acc = test_correct / len(testloader.dataset)

    y_testloss.append(epoch_test_loss)
    
    y_testacc.append(epoch_test_acc)
    
    draw_testloss(x,y_testloss)
    drwa_testacc(x,y_testacc)
    
    print('test loss is ',epoch_test_loss,'  test acc is ',epoch_test_acc)    
    torch.save(net,'./model/best.pt')