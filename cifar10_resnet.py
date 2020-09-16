import torch
import torchvision
import torchvision.transforms as transforms

import math
import time

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')


# try replacing relu with elu
torch.manual_seed(69)
default_batch=128 # no. of batches per epoch 50000/default_batch
batches_for_report=10#

transform=transforms.Compose(
   [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


trainset=torchvision.datasets.CIFAR10(root='./torchdata', train=True,
    download=True, transform=transform)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=default_batch,
    shuffle=True, num_workers=2)

testset=torchvision.datasets.CIFAR10(root='./torchdata', train=False,
    download=True, transform=transform)

testloader=torch.utils.data.DataLoader(testset, batch_size=default_batch,
    shuffle=False, num_workers=0)


classes=('plane', 'car', 'bird', 'cat', 
  'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
 
From: https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet9():
    return ResNet(BasicBlock, [1,1,1,1])

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


# enable this to use wide ResNet
wide_resnet=False
if not wide_resnet:
  net=ResNet18().to(mydevice)
else:
  # use wide residual net https://arxiv.org/abs/1605.07146
  net=torchvision.models.resnet.wide_resnet50_2().to(mydevice)


#####################################################
def verification_error_check(net):
   correct=0
   total=0
   for data in testloader:
     images,labels=data
     outputs=net(Variable(images).to(mydevice))
     _,predicted=torch.max(outputs.data,1)
     correct += (predicted==labels.to(mydevice)).sum()
     total += labels.size(0)

   return 100*correct//total
#####################################################

lambda1=0.000001
lambda2=0.001

# loss function and optimizer
import torch.optim as optim
from lbfgsnew import LBFGSNew # custom optimizer
criterion=nn.CrossEntropyLoss()
#optimizer=optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer=optim.Adam(net.parameters(), lr=0.001)
optimizer = LBFGSNew(net.parameters(), history_size=7, max_iter=2, line_search_fn=True,batch_mode=True)


load_model=False
# update from a saved model 
if load_model:
  checkpoint=torch.load('./res18.model',map_location=mydevice)
  net.load_state_dict(checkpoint['model_state_dict'])
  net.train() # initialize for training (BN,dropout)

start_time=time.time()
use_lbfgs=True
# train network
for epoch in range(20):
  running_loss=0.0
  for i,data in enumerate(trainloader,0):
    # get the inputs
    inputs,labels=data
    # wrap them in variable
    inputs,labels=Variable(inputs).to(mydevice),Variable(labels).to(mydevice)

    if not use_lbfgs:
     # zero gradients
     optimizer.zero_grad()
     # forward+backward optimize
     outputs=net(inputs)
     loss=criterion(outputs,labels)
     loss.backward()
     optimizer.step()
    else:
      if not wide_resnet:
        layer1=torch.cat([x.view(-1) for x in net.layer1.parameters()])
        layer2=torch.cat([x.view(-1) for x in net.layer2.parameters()])
        layer3=torch.cat([x.view(-1) for x in net.layer3.parameters()])
        layer4=torch.cat([x.view(-1) for x in net.layer4.parameters()])

      def closure():
        if torch.is_grad_enabled():
         optimizer.zero_grad()
        outputs=net(inputs)
        if not wide_resnet:
          l1_penalty=lambda1*(torch.norm(layer1,1)+torch.norm(layer2,1)+torch.norm(layer3,1)+torch.norm(layer4,1))
          l2_penalty=lambda2*(torch.norm(layer1,2)+torch.norm(layer2,2)+torch.norm(layer3,2)+torch.norm(layer4,2))
          loss=criterion(outputs,labels)+l1_penalty+l2_penalty
        else:
          l1_penalty=0
          l2_penalty=0
          loss=criterion(outputs,labels)
        if loss.requires_grad:
          loss.backward()
          #print('loss %f l1 %f l2 %f'%(loss,l1_penalty,l2_penalty))
        return loss
      optimizer.step(closure)
    # only for diagnostics
    outputs=net(inputs)
    loss=criterion(outputs,labels)
    running_loss +=loss.data.item()

    if math.isnan(loss.data.item()):
       print('loss became nan at %d'%i)
       break

    # print statistics
    if i%(batches_for_report) == (batches_for_report-1): # after every 'batches_for_report'
      print('%f: [%d, %5d] loss: %.5f accuracy: %.3f'%
         (time.time()-start_time,epoch+1,i+1,running_loss/batches_for_report,
         verification_error_check(net)))
      running_loss=0.0

print('Finished Training')


# save model (and other extra items)
torch.save({
            'model_state_dict':net.state_dict(),
            'epoch':epoch,
            'optimizer_state_dict':optimizer.state_dict(),
            'running_loss':running_loss,
           },'./res.model')


# whole dataset
correct=0
total=0
for data in trainloader:
   images,labels=data
   outputs=net(Variable(images).to(mydevice)).cpu()
   _,predicted=torch.max(outputs.data,1)
   total += labels.size(0)
   correct += (predicted==labels).sum()
   
print('Accuracy of the network on the %d train images: %d %%'%
    (total,100*correct//total))

correct=0
total=0
for data in testloader:
   images,labels=data
   outputs=net(Variable(images).to(mydevice)).cpu()
   _,predicted=torch.max(outputs.data,1)
   total += labels.size(0)
   correct += (predicted==labels).sum()
   
print('Accuracy of the network on the %d test images: %d %%'%
    (total,100*correct//total))


class_correct=list(0. for i in range(10))
class_total=list(0. for i in range(10))
for data in testloader:
  images,labels=data
  outputs=net(Variable(images).to(mydevice)).cpu()
  _,predicted=torch.max(outputs.data,1)
  c=(predicted==labels).squeeze()
  for i in range(4):
    label=labels[i]
    class_correct[label] += c[i]
    class_total[label] += 1

for i in range(10):
  print('Accuracy of %5s : %2d %%' %
    (classes[i],100*float(class_correct[i])/float(class_total[i])))
