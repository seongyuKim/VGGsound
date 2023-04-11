import torch
import torchvision

def MainModel(weights,nOut=1000,**kwargs):
    
    AudioResNet = torchvision.models.resnet18(num_classes=nOut,weights=weights)
    # for param in AudioResNet.parameters():
        # param.requires_grad = False
    AudioResNet.conv1 = torch.nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias=False)
    print("it's modified ResNet18 for 1 Channel")
    return AudioResNet