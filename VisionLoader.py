import os
import torch
import torchaudio
import torchvision
import torchvision .transforms as transforms
import numpy as np
import random
import math
from torch.utils.data import Dataset,DataLoader,RandomSampler
from PIL import Image
import glob
import collections
import matplotlib.pyplot as plt
import pdb

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def url2label(filelist_path):
    list_csv = open(filelist_path,'r')
    url2class_dict={}
    class_list=[]

    while True:
        line = list_csv.readline()
        if not line: break
        url_class_pair=line.split('.mp4,')
           # dict : {url : string class}   ex) {W7SCaJhNL54_000010: playing trombone}
        url2class_dict[url_class_pair[0]]=url_class_pair[1][:-1] #delete new line
        class_list.append(url_class_pair[1][:-1])
    class_list=list(set(class_list))
    class_list.sort()

    # dict : {string class: class number}   ex) {playing trombone: 12}
    class2num_dict={key:val for val,key in enumerate(class_list)}

    # dict : {url : class number}   ex) {W7SCaJhNL54_000010: 12}
    url2label_dict={url: class2num_dict[str_class] for url,str_class in url2class_dict.items()}

    #amount_of_label=len(class2num_dict.values())

    return url2label_dict , class2num_dict ,url2class_dict

def get_class_distribution(url2class_dict):
    
    class_list=list(url2class_dict.values())
    class2cnt=collections.Counter(class_list)
    classes=list(class2cnt.keys())
    cnts=list(class2cnt.values())
    plt.bar(range(len(class2cnt)),cnts,tick_label=classes)
    plt.show()

    return class2cnt





class meta_Dataset(Dataset):
    def __init__(self,audio_path,frame_path,audio_ext,image_ext,list_path,transforms=None,random_sample= False, clipping_duration=3):
        
        ## -------- make {url : class#} ----------- ##
        
        url2num_dict,class2num_dict,url2class_dict = url2label(list_path)

        self.label_dict= url2num_dict
        self.class2num_dict=class2num_dict
        self.train_url_list=list(self.label_dict.keys())
        self. train_label_list= list(self.label_dict.values())
        self.transforms=transforms
        self.audio_path=audio_path
        self.frame_path=frame_path
        self.audio_ext=audio_ext
        self.image_ext=image_ext
        self.random_sample=random_sample
        self.clipdur=clipping_duration

        '''
        ### ---- ---- ---- ---- ---- 
        ### class distribution
        ### ---- ---- ---- ---- ----
        class2cnt_dict=get_class_distribution(url2class_dict)
        f=open('/home/seon/workspace/SoundNet/output/train_class_distribution.txt','w')
        f.write(str(class2cnt_dict))
        f.close
        print(class2cnt_dict)
        '''
        #print(len(self.label_dict.values()))
        print('{:d} files of {:d} classes found'.format(len(self.label_dict),len(self.class2num_dict.values())))
        
    def __getitem__(self,index):

        #--------------------
        # for image
        #--------------------
        img_feat=[]
        #for index in indices:  
        #print(str(list(self.label_dict.keys())[index]))  
        url_path=self.frame_path + str(list(self.label_dict.keys())[index])  
        #print(url_path)       
        ## count the number of frame for the url
        frames=glob.glob('%s/*.%s'%(url_path,self.image_ext))
        frames.sort()
        tot_frames=len(frames)

            ## ==== ==== ==== ====
            ## random sampling
            ## ==== ==== ==== ====
        if self.random_sample:
            sample_frame=random.randint(1,tot_frames)
            sample_point = str(sample_frame).zfill(3)
            jpg_path='%s/%s.%s'%(url_path,sample_point,self.image_ext)
            ###print("%s is the sampled point of %s frames"%(sample_point,str(tot_frames)))

            ## ==== ==== ==== ====
            ## mid point sampling
            ## ==== ==== ==== ====
        else:
            mid_point = str(math.ceil(len(frames)/2)).zfill(3)
            jpg_path='%s/%s.%s'%(url_path,mid_point,self.image_ext)
            ###print("%s is the mid point of %s frames"%(mid_point,str(tot_frames)))
        
        

        if self.transforms is not None:
            tmp_img=Image.open(jpg_path)
            #print(tmp_img.size)
            assert tmp_img.size == (299,299) , "{} has weird shape".format(jpg_path)
            img_feat.append(self.transforms(tmp_img))  
        else:
            img_feat.append(Image.open(jpg_path))
            
        img_feat=np.stack(img_feat,axis=0)
        '''
        toimg=torchvision.transforms.ToPILImage()
        tmpimg=torch.Tensor(img_feat).squeeze()
        tmpimg=toimg(tmpimg)
        tmpimg.save('loaded_data_%s/image.jpg'%(list(self.label_dict.keys())[index]))
        '''
        #print(img_feat.shape)
        
        '''
        #--------------------
        # for audio
        #--------------------
        wav_path = "%s%s.%s" %(self.audio_path , str(list(self.label_dict.keys())[index]), self.audio_ext)  
        #print(wav_path)
        waveform, sample_rate = torchaudio.load(wav_path)

        totlen=waveform.shape[-1]
        #print(totlen)
            ## ==== ==== ==== ====
            ## random sampling
            ## ==== ==== ==== ====
        if self.random_sample:
            frac= sample_frame/tot_frames
            random_point_a = int(totlen*frac)
            start = int(random_point_a - int((self.clipdur/2)*sample_rate))
            end   = int(random_point_a + int((self.clipdur/2)*sample_rate))
            ###print("%s is the sampled point of %s samples"%(random_point_a,totlen))
            
            ## ==== ==== ==== ====
            ## mid point sampling
            ## ==== ==== ==== ====
        else:
            mid_point_a = math.ceil((totlen/2))
            start = int(mid_point_a - int((self.clipdur/2)*sample_rate))
            end   = int(mid_point_a + int((self.clipdur/2)*sample_rate))
            ###print("%s is the middle point of %s samples"%(mid_point_a,totlen))
            
        keep_samples= int(sample_rate * self.clipdur)  
        if (start <= 0) :
            clipped_waveform=waveform[:,:keep_samples]
        elif (end >= totlen):
            clipped_waveform= waveform[:, totlen-keep_samples:]
        else:
            clipped_waveform = waveform[:,start:end]

        #torchaudio.save('loaded_data_%s.audio.wav'%(list(self.label_dict.keys())[index],index),clipped_waveform,sample_rate)
        '''
        


        
        #--------------------
        # return values
        #--------------------
        image = torch.FloatTensor(img_feat)
        #audio = torch.FloatTensor(clipped_waveform)
        label = self.train_label_list[index]
        
        return image, label
        
    def __len__(self):
        return len(self.label_dict)

# ---- ---- ---- ---- ---- 
# is it necessary? 
# ---- ---- ---- ---- ---- 
'''
class test_dataset_loader(Dataset):
    def __init__():
        pass
    def __getitem__():
        pass
    def __len__():
        pass

def get_data_loader():
    pass
'''

# ==== ==== ==== ==== ==== ==== ==== 
# Sampler
# ==== ==== ==== ==== ==== ==== ====
'''
class train_sampler(torch.utils.data.Sampler):
    def __init__():
        pass
    def __iter__():
        pass
    def __len__():
        pass
'''

def get_data_loader(batch_size,nDataLoaderThread,audio_path,frame_path,audio_ext,image_ext,list_path,transforms,random_sample,clipping_duration,**kwargs):
    train_dataset = meta_Dataset(audio_path=audio_path,frame_path=frame_path,audio_ext=audio_ext,image_ext=image_ext,list_path=list_path,transforms=transforms,random_sample=random_sample,clipping_duration=clipping_duration)

    train_sampler = RandomSampler(train_dataset)

    train_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    return train_loader


if __name__ == "__main__":
    audio_path = "/home/seon/workspace/SoundNet/VGGSound_final/audio/"
    frame_path = "/home/seon/workspace/SoundNet/VGGSound_final/frames/"
    audio_ext = "wav"
    image_ext = "jpg"
    train_list_path = "/home/seon/workspace/SoundNet/list/train.csv"
    test_list_path  = "/home/seon/workspace/SoundNet/list/test.csv"
    
    train_transform= transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(256),
        transforms.RandomCrop([224,224]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    

    #dataset = meta_Dataset(audio_path,frame_path,audio_ext,image_ext,list_path=train_list_path,random_sample=False,clipping_duration=3,transforms=train_transform)
    #dataloader=DataLoader(dataset,batch_size=2,sampler=RandomSampler(dataset))
    dataloader=get_data_loader(batch_size=1,nDataLoaderThread=6,audio_path=audio_path,frame_path=frame_path,audio_ext=audio_ext,image_ext=image_ext,list_path=train_list_path,transforms=train_transform,random_sample=True,clipping_duration=3)

    it=iter(dataloader)
    img,label=next(it)
    #img=img.squeeze(dim=1)
    print(img.shape)
    img=img.squeeze(0)
    img=img.squeeze(0)
    print(img.shape)
    img=img.numpy().astype(np.uint8)
    img=img.transpose(1,2,0)
    print(img.shape)
    new_img=Image.fromarray(img)
    

    new_img.save("/home/seon/workspace/SoundNet/toy/afterloader.jpg")
    
    #print(label)

    #label=dict(collections.Counter(label.tolist()))
    #label=sorted(label.items(),key= lambda item:item[1])
    #print(label)

   
    
