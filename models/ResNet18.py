import torchvision

def MainModel(pretrained,nOut=512,**kwargs):
    
    
    return torchvision.models.resnet18(num_classes=nOut,pretrained=pretrained)