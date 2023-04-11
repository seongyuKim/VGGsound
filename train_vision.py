import sys
import time
import os
import argparse
import glob
import datetime
from torch.utils import *

from Soundnet_vision import *
from VisionLoader import *

import torchvision.transforms as transforms


## ==== ==== ==== ==== ==== ==== ==== 
##  Parse arguments
## ==== ==== ==== ==== ==== ==== ==== 

parser=argparse.ArgumentParser(description = "VGGsound training");

## Data Loader
parser.add_argument('--batch_size',         type=int,   default=64,  help='Batch size');
parser.add_argument('--nDataLoaderThread',  type=int,   default=6,    help='Number of loader threads');
parser.add_argument('--clipping_duration',  type=int,   default=3,    help='How long to cut audio clips');
parser.add_argument('--random_sample',      type=bool,  default=False, help='random sampling(True) or middle sampling(False)');
parser.add_argument('--new_sample_rate',    type=int,   default=16000, help='resampling rate for audio wav file');

## Training and evaluation data
parser.add_argument('--train_list_path',    type=str,   default="/home/seon/workspace/SoundNet/list/train.csv", help='Absolute path to the train list');
parser.add_argument('--test_list_path',     type=str,   default="/home/seon/workspace/SoundNet/list/test.csv",  help='Absloute path to the test list');
parser.add_argument('--frame_path',         type=str,   default="/home/seon/workspace/SoundNet/VGGSound_final/frames/", help='path to the VGGsound frame data');
parser.add_argument('--audio_path',         type=str,   default="/home/seon/workspace/SoundNet/VGGSound_final/audio/",  help='path to the VGGsound audio data');
parser.add_argument('--image_ext',          type=str,   default="jpg",  help='extension of frame data');
parser.add_argument('--audio_ext',          type=str,   default="wav",  help='extension of audio data');
parser.add_argument('--get_pretrained',     type=bool,  default= False,     help='whether load pretrained resnet or not');

## Training details
parser.add_argument('--test_interval',      type=int,   default=5,      help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',          type=int,   default=100,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',          type=str,   default='CEloss', help='loss function');


## Optimizer
parser.add_argument('--optimizer',          type=str,   default='adam', help='optimizer');
parser.add_argument('--scheduler',          type=str,   default='ExponentialLR', help='learning rate scheduler');
parser.add_argument('--lr',                 type=float, default=1e-3, help='learning rate');
parser.add_argument('--lr_decay',           type=float, default=0.99, help='Learningrate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',      type=float, default=1e-4, help='weight decay in the optimizer');

## Loss functions

## Load and save
parser.add_argument('--initial_model',      type=str,   default="", help='Initial model weights');
parser.add_argument('--save_path',          type=str,   default="exps/exp1", help='Path for model and logs');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true',   help='Eval only')
parser.add_argument('--output',         type=str,   default="",     help='Save a log of output to this file name');

## Model definition
parser.add_argument('--vision_model',              type=str,   default='ResNet18', help='Name of vision model definition');
parser.add_argument('--vision_loss',               type=str,   default='CEloss',   help='loss function for vision network');
parser.add_argument('--audio_model',              type=str,   default='ResNet18_Audio', help='Name of audio model definition');
parser.add_argument('--audio_loss',                 type=str,   default='CEloss', help='loss function for audio network');
parser.add_argument('--nOut',               type=int,   default=1000,        help='Embedding size in the last FC layer');
parser.add_argument('--nClasses',           type=int,   default=309,        help='number of classes in VGGsound dataset')

## Training
parser.add_argument('--gpu',                type=int,   default=0,          help='GPU index');
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--model_index',        type=int,   default=0,          help='choose model[vision(0),audio(1),multimodal(2)]')

## additional arguments
parser.add_argument('--augment',        type=str,   default="",  help='data augmentation while training');
parser.add_argument('--rotate_angle',   type=int,   default=30,  help='angle used for RandomRotation of data augmentation');


args = parser.parse_args();


## ==== ==== ==== ==== ==== ==== ==== 
##  Trainer script
## ==== ==== ==== ==== ==== ==== ==== 

def main_worker(args):
    
    ## Load models 
    """
    need to modify to/after combine both networks
    """
    #ChosenNet=AudioNet(**vars(args)).cuda()
    
    
    if args.model_index == 0 :
        print('Choosing VisionNet!')
        ChosenNet=VisionNet(**vars(args)).cuda()
    elif args.model_index == 1 :
        print('Choosing AudioNet!')
        ChosenNet=AudioNet(**vars(args)).cuda()
    elif args.model_index == 2:
        print('Choosing MultimodalNet!')
        #ChosenNet= MultimodalNet(**vars(args)).cuda()
    else:
        assert args.model_index <= 2, 'wrong model index, choose between 0~2'
    
    

    it = 1

    ## Input transformations for training
        ## Input transformations for training
    if (args.augment == ""):
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize(256),
            transforms.RandomCrop([224,224]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        print("Data Augmentation will be done")
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize(256),
            transforms.RandomCrop([224,224]),
            ### augmentation more
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(args.rotate_angle),
            ### augmentation much more
            transforms.ColorJitter(brightness=(0,1),saturation=(0,1),hue=0),
            #transforms.RandomErasing(p=0.1,scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    ## Input transformations for evaluation
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop([224,224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])    

    
    ## Initialize trainer and data loader
    trainloader = get_data_loader(transforms= train_transform, list_path= args.train_list_path,**vars(args));

    trainer = ModelTrainer(ChosenNet,**vars(args))
    """
    <-- need to modify after combine both networks
    """

    ## Load model weights
    modelfiles = glob.glob('{}/model0*.model'.format(args.save_path))
    modelfiles.sort()

    ## If the target directory already exists, start from the existing file
    if len(modelfiles) >= 1: ### (already implemented before)
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1 ###(get number int(0*) from ~/model0*.model)
    elif (args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))

    ## If the current iteration is not 1, update the scheduler
    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## print total number of model parameters
    pytorch_total_params = sum(p.numel() for p in ChosenNet.__S__.parameters())
    print('Total model paramters: {:,}'.format(pytorch_total_params)) #{:,} split integer numbers with commas separating groups of thousands.

    ## Evaluation code
    if args.eval == True:
        acc= trainer.evaluateFromList(test_path=args.test_list_path,transform=test_transform, **vars(args))
        print('acc {:.7f}'.format(acc))

        if args.output != '':
            with open(args.output,'w') as f:
                f.write('{}',format(acc))
        
        quit();
        
    ## Write args to socrefile for training
    scorefile = open(args.save_path+"/scores.txt","a+");
    strtime= datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scorefile.write('{}\n{}\n'.format(strtime,args))
    scorefile.flush()

    ## Core training script
    for it in range(it,args.max_epoch+1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        print(time.strftime("%Y-%m-%d %H:%M:%S"),it,"Training epoch {:d} with LR{:.5f}".format(it,max(clr)));

        loss = trainer.train_network(trainloader);

        if it % args.test_interval==0:
            acc = trainer.evaluateFromList(test_path=args.test_list_path,transform=test_transform, **vars(args))
            print("IT {:d}, accuracy {:.5f}".format(it, acc));
            scorefile.write("IT {:d}, accuracy {:.5f}".format(it, acc));
            trainer.saveParameters(args.save_path+"/model{:09d}.model".format(it));
        
        print(time.strftime("%Y-%m-%d %H:%M:%S"),"TLOSS {:5f}".format(loss))
        scorefile.write("IT {:d}, TLOSS {:.5f} with LR ".format(it,loss)+str(clr)+"\n")
        scorefile.flush()
    scorefile.close()
## ==== ==== ==== ==== ==== ==== ==== 
## Main function
## ==== ==== ==== ==== ==== ==== ==== 

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(args.gpu)

    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
    main_worker(args)

if __name__ == '__main__':
    main()

