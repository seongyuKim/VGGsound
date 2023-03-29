import torch
import torchvision

def MainModel(pretrained,nOut=512,**kwargs):
    
    AudioResNet = torchvision.models.resnet18(num_classes=nOut,pretrained=pretrained)
    AudioResNet.conv1 = torch.nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias=False)
    return AudioResNet