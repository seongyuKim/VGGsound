import torch
import torch.nn as nn

class SoundNet(nn.Module):
    def __init__(self, nOut):
        super(SoundNet,self).__init__()
        
        self.nOut= nOut
        
        ## ===== ===== ===== ===== ===== 
        ##  model definition
        ## ===== ===== ===== ===== =====
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=64,stride=2,padding=32),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool1d(kernel_size=8,stride=8,padding=0)##논문실수??
        self.conv2 = nn.Sequential(
            nn.Conv1d(16,32,32,stride=2,padding=16),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool1d(kernel_size=8,stride=8,padding=0)
        self.conv3 = nn.Sequential(
            nn.Conv1d(32,64,16,stride=2,padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(64,128,8,stride=2,padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(128,256,4,stride=2,padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool1d(kernel_size=4,stride=4,padding=0)
        self.conv6 = nn.Sequential(
            nn.Conv1d(256,512,4,stride=2,padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(512,1024,4,stride=2,padding=2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        
        )
        self.conv8 = nn.Conv1d(1024,self.nOut,3,stride=2,padding=0)
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,16, (64,1), (2,1), (32,0), bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8,1), (8,1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (32,1), (2,1), (16,0), bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8,1),(8,1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (16,1), (2,1), (8,0), bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, (8,1), (2,1), (4,0), bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, (4,1),(2,1),(2,0), bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4,1),(4,1))
        ) # difference here (0.24751323, 0.2474), padding error has beed debuged
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, (4,1), (2,1), (2,0), bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 1024, (4,1), (2,1), (2,0), bias=True),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        #self.conv8 = nn.Conv1d(1024,self.num_classes,8,stride=2,padding=0)
        self.conv8 = nn.Conv2d(1024, self.num_classes, (8,1), (2,1), (0,0), bias=True)
        '''
    def forward(self,x):
        x=self.conv1(x)
        #print("After Conv1d: ", x.shape)
        x=self.pool1(x)
        #print("After Conv1d: ", x.shape)
        x=self.conv2(x)
        x=self.pool2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.pool5(x)
        x=self.conv6(x)
        x=self.conv7(x)
        #print("BEFORE CONV8: ",x.shape)
        x=self.conv8(x)

        return x


def MainModel(nOut,**kwargs):
    return SoundNet(nOut=nOut) 





