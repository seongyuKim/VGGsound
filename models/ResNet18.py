import torchvision

def MainModel(weights,nOut,**kwargs):
    
    
    return torchvision.models.resnet18(num_classes=nOut,weights=weights)