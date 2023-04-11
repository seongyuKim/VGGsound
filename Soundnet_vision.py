import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy,math,pdb,sys
import time,importlib
import VisionLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

class VisionNet(nn.Module):
    def __init__(self,vision_model,optimizer,vision_loss,get_pretrained,**kwargs):
        super(VisionNet,self).__init__();
        ##__S__ is the embedding model
        VisionNetModel=importlib.import_module('models.'+ vision_model).__getattribute__('MainModel')
        if get_pretrained==True:
            self.__S__ = VisionNetModel(**kwargs,weights='IMAGENET1K_V1')
            print("pretrained ResNet is loaded")
        else:
            self.__S__ = VisionNetModel(**kwargs,weights=None)
            print("not pretrained model is loaded")

        ##__L__ is the classifier plus the loss function
        LossFunction = importlib.import_module('loss.'+ vision_loss).__getattribute__('LossFunction')
        print("it's for VisionNet")
        self.__L__ = LossFunction(**kwargs)
    
    def forward(self,data,label=None):
        # data = data.reshape ??? let's see later
        #data    = data.reshape(-1,data.size()[-3],data.size()[-2],data.size()[-1])
        output=self.__S__.forward(data)
        

        if label== None:
             return output
        else:
            # output=output.reshape ??? let's see later 
            #print("!!!",output.shape)
            nloss=self.__L__.forward(output,label)
            return nloss
            

class AudioNet(nn.Module):
    def __init__(self,audio_model,optimizer,audio_loss,get_pretrained,**kwargs):
        super(AudioNet,self).__init__();
        ##__S__ is the embedding model
        AudioNetModel=importlib.import_module('models.'+ audio_model).__getattribute__('MainModel')
        if get_pretrained == True:
            self.__S__ = AudioNetModel(**kwargs,weights='IMAGENET1K_V1')
            print("pretrained ResNet is loaded")
        else:
            self.__S__ = AudioNetModel(**kwargs,weights=None)
            print("not pretrained model is loaded")

        ##__L__ is the classifier plus the loss function
        LossFunction = importlib.import_module('loss.'+ audio_loss).__getattribute__('LossFunction')
        print("it's for AudioNet")
        self.__L__ = LossFunction(**kwargs)
    
    def forward(self,data,label=None):
        # data = data.reshape ??? let's see later
        output=self.__S__.forward(data)
        #print("AFTER SOUNDNET:",output.shape) [320,nOut,1]

        if label== None:
             return output
        else:
            #print("in AudioNet", output.shape)
            #output=output.squeeze(2) 
            nloss=self.__L__.forward(output,label)
            return nloss


class ModelTrainer(object):
    def __init__(self, embed_model,optimizer, scheduler, mixedprec,**kwargs):
        self.__model__=embed_model

        ## Optimizer(e.g. Adam or SGD)
        Optimizer=importlib.import_module('optimizer.'+ optimizer).__getattribute__('Optimizer')
        self.__optimizer__=Optimizer(self.__model__.parameters(),**kwargs)

        ## Learning Rate Scheduler
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__,self.lr_step=Scheduler(self.__optimizer__,**kwargs)

        ## for mixed_precision training
        self.scaler=GradScaler()
        self.mixedprec=mixedprec

        assert self.lr_step in ['epoch','iteration']

    # ## ===== ===== ===== ===== ===== 
    # ## Train Network
    # ## ===== ===== ===== ===== ===== 
    def train_network(self,loader):

        self.__model__.train();
        stepsize = loader.batch_size;

        counter = 0
        index   = 0
        loss    = 0

        with tqdm(loader,unit='batch') as tepoch:
            for ii in tepoch: ### seperate data into each modalities?
                Vdata,label = ii[0],ii[1]
                
                tepoch.total = tepoch.__len__()
                
                #adjusting data into proper shape
                #print('before: ',Vdata.shape)
                Vdata=Vdata.reshape(-1,Vdata.size()[-3],Vdata.size()[-2],Vdata.size()[-1])
                
                #print('after: ',Vdata.shape)
                #Adata=Adata.squeeze(dim=1)

                ##Reset gradients
                self.__model__.zero_grad();

                ##Forward and Backward passes
                # Adata=Adata.unsqueeze(1)
                #print(Adata.shape)
                
                #print(label.shape)
                if self.mixedprec:
                    with autocast():
                        nloss=self.__model__(Vdata.cuda(),label.cuda())
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()
                else:
                    nloss=self.__model__(Vdata.cuda(),label.cuda())
                    nloss.backward();
                    self.__optimizer__.step();

                loss+= nloss.detach().cpu().item();
                counter+=1;
                index+=stepsize;

                # Print statisctics to progress bar (on right side of progress bar)
                tepoch.set_postfix(loss=loss/counter)

                if self.lr_step == 'iteration' : self.__scheduler__.step()

            if self.lr_step == 'epoch': self.__scheduler__.step()

        return (loss/counter);

    ## ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== 

    def evaluateFromList(self,test_path,audio_path,frame_path,audio_ext,image_ext,random_sample,clipping_duration,nDataLoaderThread, transform,batch_size,**kwargs):
        self.__model__.eval();

        feats={}
        ## Define test data loader
        test_dataset =  VisionLoader.meta_Dataset(audio_path,frame_path,audio_ext,image_ext,test_path,transform,random_sample=random_sample,clipping_duration=clipping_duration)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False
            )
        ep_acc=0
        ep_cnt=0
        softmax=torch.nn.Softmax(dim=1)
        for Vdata,label in tqdm(test_loader):
            
            ##data reshape
            Vdata=Vdata.reshape(-1,Vdata.size()[-3],Vdata.size()[-2],Vdata.size()[-1]) #[batch,3,244,244]
            
            #Adata=Adata.squeeze(dim=1) #[batch,48000]
            # print(Adata.shape)
            # Adata=Adata.unsqueeze(1)
            output= self.__model__(Vdata.cuda())
            #output=torch.nn.Softmax(output,dim=1)

            ## Find predicted label
            #output=output.squeeze(2)
            output = self.__model__.__L__.fc_layer(output)
            #print("raw output: ",output)
            #print("raw argmax: ",torch.argmax(output,dim=1))
            output=softmax(output)
            #print("softmax output: ",output)
            #print("eval output shape: ",output.shape)
            label_pred = torch.argmax(output,dim=1) 
            #print("label: ",label)
            #print("label_pred: ",label_pred)
            ep_acc  += (label_pred.detach().cpu() ==label).sum().item()
            #ep_loss += self.__model__(output,label) * len(Vdata)
            ep_cnt  += len(Vdata)
            #print(ep_acc)
            #print(ep_cnt)

        acc= ep_acc / ep_cnt
        return acc

            


            

    ## ===== ===== ===== ===== ===== ===== =====
    ## Save Parameters
    ## ===== ===== ===== ===== ===== ===== =====
    def saveParameters(self,path):

        torch.save(self.__model__.state_dict(),path);

    ## ===== ===== ===== ===== ===== ===== ===== 
    ## Load Parameters
    ## ===== ===== ===== ===== ===== ===== =====
    def loadParameters(self,path):

        self_state = self.__model__.state_dict(); # now model
        loaded_state=torch.load(path); ## code for load model
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                # if name not in self_state:
                print("{} is not in the model.".format(origname))
                continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {},model: {}.loaded: {}".format(origname,self_state[name].size(),loaded_state[origname.size()]))
                continue
            self_state[name].copy_(param); # want to copy params for loaded model to the now model

        