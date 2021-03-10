import torch
import torch.nn as nn
import torch.nn.functional as F
from utility.ResBlock import ResBlock

class ResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__() 
        self.conv1_1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,padding_mode='reflect')
        self.bn1_1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(3,2)
        self.conv2_1 = ResBlock(nn.Conv2d(64,64,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(64), \
                       nn.ReLU(), \
                       nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(64), \
                       nn.ReLU(), \
                       nn.Conv2d(64,256,kernel_size=1,stride=1),\
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), res_transform=nn.Conv2d(64,256,kernel_size=1,stride=1))
        self.conv2_2 = ResBlock(nn.Conv2d(256,64,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(64), \
                       nn.ReLU(), \
                       nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(64), \
                       nn.ReLU(), \
                       nn.Conv2d(64,256,kernel_size=1,stride=1),\
                       nn.BatchNorm2d(256), \
                       nn.ReLU())
        self.conv2_3 = ResBlock(nn.Conv2d(256,64,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(64), \
                       nn.ReLU(), \
                       nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(64), \
                       nn.ReLU(), \
                       nn.Conv2d(64,256,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU())
        self.conv3_1 = ResBlock(nn.Conv2d(256,128,kernel_size=1,stride=2), \
                       nn.BatchNorm2d(128), \
                       nn.ReLU(), \
                       nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(128), \
                       nn.ReLU(), \
                       nn.Conv2d(128,512,kernel_size=1), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU(), res_transform=nn.Conv2d(256,512,kernel_size=1,stride=2))
        self.conv3_2 = ResBlock(nn.Conv2d(512,128,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(128), \
                       nn.ReLU(), \
                       nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(128), \
                       nn.ReLU(), \
                       nn.Conv2d(128,512,kernel_size=1), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU())
        self.conv3_3 = ResBlock(nn.Conv2d(512,128,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(128), \
                       nn.ReLU(), \
                       nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(128), \
                       nn.ReLU(), \
                       nn.Conv2d(128,512,kernel_size=1), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU())
        self.conv3_4 = ResBlock(nn.Conv2d(512,128,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(128), \
                       nn.ReLU(), \
                       nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(128), \
                       nn.ReLU(), \
                       nn.Conv2d(128,512,kernel_size=1), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU())
        self.conv4_1 = ResBlock(nn.Conv2d(512,256,kernel_size=1,stride=2), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,1024,kernel_size=1,stride=1),\
                       nn.BatchNorm2d(1024), \
                       nn.ReLU(), \
                       res_transform=nn.Conv2d(512,1024,kernel_size=1,stride=2))
        self.conv4_2 = ResBlock(nn.Conv2d(1024,256,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,1024,kernel_size=1,stride=1),\
                       nn.BatchNorm2d(1024), \
                       nn.ReLU())
        self.conv4_3 = ResBlock(nn.Conv2d(1024,256,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,1024,kernel_size=1,stride=1),\
                       nn.BatchNorm2d(1024), \
                       nn.ReLU())
        self.conv4_4 = ResBlock(nn.Conv2d(1024,256,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,1024,kernel_size=1,stride=1),\
                       nn.BatchNorm2d(1024), \
                       nn.ReLU())
        self.conv4_5 = ResBlock(nn.Conv2d(1024,256,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,1024,kernel_size=1,stride=1),\
                       nn.BatchNorm2d(1024), \
                       nn.ReLU())
        self.conv4_6 = ResBlock(nn.Conv2d(1024,256,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(256), \
                       nn.ReLU(), \
                       nn.Conv2d(256,1024,kernel_size=1,stride=1),\
                       nn.BatchNorm2d(1024), \
                       nn.ReLU())
        self.conv5_1 = ResBlock(nn.Conv2d(1024,512,kernel_size=1,stride=2), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU(), \
                       nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU(), \
                       nn.Conv2d(512,2048,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(2048), \
                       nn.ReLU(),
                       res_transform=nn.Conv2d(1024,2048,kernel_size=1,stride=2))
        self.conv5_2 = ResBlock(nn.Conv2d(2048,512,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU(), \
                       nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU(), \
                       nn.Conv2d(512,2048,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(2048), \
                       nn.ReLU())
        self.conv5_3 = ResBlock(nn.Conv2d(2048,512,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU(), \
                       nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,padding_mode='reflect'), \
                       nn.BatchNorm2d(512), \
                       nn.ReLU(), \
                       nn.Conv2d(512,2048,kernel_size=1,stride=1), \
                       nn.BatchNorm2d(2048), \
                       nn.ReLU())
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048,n_classes)
         
    def forward(self,x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.max_pool(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.avg_pool(x).view(-1,2048)
        x = self.fc(x)
        return x
