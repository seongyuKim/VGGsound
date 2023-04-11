import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

class LossFunction(nn.Module):
	def __init__(self,**kwargs):
		super(LossFunction,self).__init__()
		
		self.criterion=torch.nn.KLDivLoss(reduction='batchmean')
		#self.fc = nn.Linear(nOut,nClasses)
		
		print('Initialised KL divergence Loss')
		
	def forward(self,vision_feature,audio_feature):
		#audio_distribution=self.fc(f_audio)
		
		vision_distribution=F.softmax(vision_feature,dim=1)
		audio_distribution=F.log_softmax(audio_feature,dim=1)
		
		nloss = self.criterion(audio_distribution,vision_distribution)
		
		return nloss