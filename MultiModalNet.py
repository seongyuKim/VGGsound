import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy,math,pdb,sys
import time,importlib
import MMLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import sklearn.metrics

class VisionNet(nn.Module):
    def __init__(self,vision_model,optimizer,vision_loss,get_pretrained,**kwargs):
        super(VisionNet,self).__init__();
        ##__S__ is the embedding model
        VisionNetModel=importlib.import_module('models.'+ vision_model).__getattribute__('MainModel')
        if get_pretrained:
            self.__S__ = VisionNetModel(**kwargs,weights='IMAGENET1K_V1')
            print("pretrained model is loaded")
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
        if get_pretrained:
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
        
class NOfusionMM(nn.Module):
    def __init__(self,vision_model,vision_loss,audio_model,audio_loss,get_pretrained,trainfunc,**kwargs):
        super(NOfusionMM,self).__init__();

        V=VisionNet(vision_model=vision_model,vision_loss=vision_loss,get_pretrained=get_pretrained,**kwargs)
        A=AudioNet(audio_model=audio_model,audio_loss=audio_loss,get_pretrained=False,**kwargs)
        self.__V__=V
        self.__A__=A
        self.__VS__ = V.__S__
        ### freeze vision extractor
        for param in self.__VS__.parameters():
            param.requires_grad =False
        self.__VL__ = V.__L__
        self.__S__ = A.__S__
        self.__L__ = A.__L__

        LossFunction = importlib.import_module('loss.'+ trainfunc).__getattribute__('LossFunction')
        print("it's Loss for Distribution")
        self.__D__ = LossFunction(**kwargs)


    def forward(self, vision_data,audio_data,label=None):
        
        vision_feature = self.__VS__.forward(vision_data)
        audio_feature = self.__S__.forward(audio_data)
        kldloss=self.__D__(vision_feature,audio_feature)
        return kldloss
    
class ConcatMM(nn.Module):
    def __init__(self,V_initial_model,vision_model,vision_loss,A_initial_model,audio_model,audio_loss,get_pretrained, trainfunc,**kwargs):
        super(ConcatMM,self).__init__();
        print(get_pretrained)
        V = VisionNet(vision_model=vision_model,vision_loss=vision_loss,get_pretrained=get_pretrained,**kwargs)
        A = AudioNet(audio_model=audio_model,audio_loss=audio_loss,get_pretrained=False,**kwargs)

        if V_initial_model != "":
            print("model from {} loaded".format(V_initial_model))
            V.load_state_dict(torch.load(V_initial_model))
        if A_initial_model != "":
            print("model from {} loaded".format(A_initial_model))
            A.load_state_dict(torch.load(A_initial_model))

        self.__V__=V
        self.__A__=A
        self.__VS__ = V.__S__
        ### freeze vision extractor
        for param in self.__VS__.parameters():
            param.requires_grad =False
        self.__VL__ = V.__L__
        self.__S__ = A.__S__
        self.__L__ = A.__L__

        LossFunction = importlib.import_module('loss.'+ trainfunc).__getattribute__('LossFunction')
        print("it's Loss for Multimodal Concat")
        self.__D__ = LossFunction(**kwargs)

    def forward(self, vision_data,audio_data,label=None):
        
        vision_feature = self.__VS__.forward(vision_data)
        audio_feature = self.__S__.forward(audio_data)
        
        #print(vision_feature.shape)
        #print(audio_feature.shape)

        concat_feature = torch.cat([vision_feature,audio_feature],dim=1)

        if label== None:
             return concat_feature
        else:
            nloss=self.__D__.forward(concat_feature,label)
            return nloss
        
class DLConcatMM(nn.Module):
    def __init__(self,V_initial_model,vision_model,vision_loss,A_initial_model,audio_model,audio_loss,get_pretrained, trainfunc,**kwargs):
        super(DLConcatMM,self).__init__();

        print(get_pretrained)
        V = VisionNet(vision_model=vision_model,vision_loss=vision_loss,get_pretrained=False,**kwargs)
        A = AudioNet(audio_model=audio_model,audio_loss=audio_loss,get_pretrained=False,**kwargs)

        if V_initial_model != "":
            print("model from {} loaded".format(V_initial_model))
            V.load_state_dict(torch.load(V_initial_model))
        if A_initial_model != "":
            print("model from {} loaded".format(A_initial_model))
            A.load_state_dict(torch.load(A_initial_model))

        self.__V__=V
        self.__A__=A
        self.__VS__ = V.__S__
        ### freeze vision extractor
        for param in self.__VS__.parameters():
            param.requires_grad =False
        self.__VL__ = V.__L__
        self.__S__ = A.__S__
        self.__AL__ = A.__L__

        LossFunction = importlib.import_module('loss.'+ trainfunc).__getattribute__('LossFunction')
        print("it's Loss for Multimodal Concat")
        self.__D__ = LossFunction(**kwargs)


    def forward(self, vision_data,audio_data,label=None):
        
        vision_feature = self.__VS__.forward(vision_data)
        audio_feature = self.__S__.forward(audio_data)
        
        #print(vision_feature.shape)
        #print(audio_feature.shape)

        concat_feature = torch.cat([vision_feature,audio_feature],dim=1)

        if label== None:
             return concat_feature
        else:
            V_LOSS = self.__VL__.forward(vision_feature,label)
            A_LOSS = self.__AL__.forward(audio_feature,label)
            C_LOSS = self.__D__.forward(concat_feature,label)

            nloss = C_LOSS + 0.1*V_LOSS + 0.1*A_LOSS
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
                Vdata,Adata,label = ii[0],ii[1],ii[2]
                
                tepoch.total = tepoch.__len__()

                ##Reset gradients
                self.__model__.zero_grad();
                
                #adjusting data into proper shape
                #print('before: ',Vdata.shape)
                
                ##Forward and Backward passes
                Vdata=Vdata.reshape(-1,Vdata.size()[-3],Vdata.size()[-2],Vdata.size()[-1])
                Adata=Adata.unsqueeze(1)
                # Adata= numpy.repeat(Adata,3,axis=1)
                #print(Adata.shape)
                
                #print(label.shape)
                if self.mixedprec:
                    with autocast():
                        nloss=self.__model__(Vdata.cuda(),Adata.cuda(),label.cuda())
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()
                else:
                    nloss=self.__model__(Vdata.cuda(),Adata.cuda(),label.cuda())
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

    def evaluateFromList(self,test_path,audio_path,frame_path,audio_ext,image_ext,random_sample,clipping_duration,nDataLoaderThread, transform,batch_size,nClasses,**kwargs):
        self.__model__.eval();

        feats={}
        ## Define test data loader
        test_dataset =  MMLoader.meta_Dataset(audio_path,frame_path,audio_ext,image_ext,test_path,transform,random_sample=True,clipping_duration=clipping_duration)
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
        
        mAP=0
        test_len=len(test_dataset)
        pred_array=numpy.zeros([test_len,nClasses])
        gt_array=numpy.zeros([test_len,nClasses])

        for count,datas in enumerate(tqdm(test_loader)):
            Vdata=datas[0]
            Adata=datas[1]
            label=datas[2]
            ##data reshape
            Vdata=Vdata.reshape(-1,Vdata.size()[-3],Vdata.size()[-2],Vdata.size()[-1]) #[batch,3,244,244]
            
            #Adata=Adata.squeeze(dim=1) #[batch,48000]
            # print(Adata.shape)
            Adata=Adata.unsqueeze(1)
            # Adata= numpy.repeat(Adata,3,axis=1)
            output= self.__model__(Vdata.cuda(),Adata.cuda())
            #output=torch.nn.Softmax(output,dim=1)

            ## Find predicted label
            #output=output.squeeze(2)
            output = self.__model__.__D__.fc_layer(output)
            output=softmax(output)
            #label_pred = torch.argmax(output,dim=1) 
            

            ## ===== ===== ===== ===== ===== 
            ## accuracy
            ## ===== ===== ===== ===== =====
            # ep_acc  += (label_pred.detach().cpu() ==label).sum().item()
            # ep_cnt  += len(Adata)
            
            ## ===== ===== ===== ===== ===== 
            ## mAP
            ## ===== ===== ===== ===== =====
            # label=F.one_hot(label,num_classes=nClasses)
            #label_pred=F.one_hot(label_pred,num_classes=nClasses)
            
            pred_array[count,:]=output.detach().cpu().numpy()
            gt_array[count,int(label)]=1
            
        stats = calculate_stats(pred_array,gt_array)
        mAP = numpy.mean([stat['AP'] for stat in stats])
        return mAP

            


            

    ## ===== ===== ===== ===== ===== ===== =====
    ## Save Parameters
    ## ===== ===== ===== ===== ===== ===== =====
    def saveParameters(self,path):

        torch.save(self.__model__.__A__.state_dict(),path);

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

### ===== ===== ===== ===== =====
### for calculating mAP
### ===== ===== ===== ===== =====
def calculate_stats(output, target):
    
    """Calculate statistics including mAP, AUC, etc.
    Args:
    output: 2d array, (samples_num, classes_num)
    target: 2d array, (samples_num, classes_num)
    Returns:
    stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = sklearn.metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = sklearn.metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = sklearn.metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = sklearn.metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc}
        stats.append(dict)

    return stats

        