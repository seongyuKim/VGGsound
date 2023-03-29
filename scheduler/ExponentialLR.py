#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):

	sche_fn = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

	lr_step = 'epoch'

	print('Initialised Exponential LR scheduler')

	return sche_fn, lr_step
