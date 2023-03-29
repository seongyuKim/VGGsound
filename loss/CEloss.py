import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

class LossFunction(nn.Module):
    def __init__(self,nOut,nClasses,**kwargs):
        super(LossFunction,self).__init__()
		
        self.criterion=torch.nn.CrossEntropyLoss()
        self.fc_layer = nn.Linear(nOut,nClasses)
        print('==> Initialised CE Loss.')
		
    def forward(self,x,label=None):
        #print("x",x.shape)
        x=self.fc_layer(x)
        loss=self.criterion(x,label)
        return loss