# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 05:56:12 2019

@author: kb
"""

import torch
from data_prep_utils import *
from data_prep_main import *
import torch.optim as optim
import numpy as np
from datetime import datetime
import time
import os
from model_main import *

####################### argument

test_id = 3 # id of test subject, for cross-validation
data_cfg = 0 # 0 for 14 class, 1 for 28
class_num = 14
batch_size = 32
learning_rate = 1e-3
num_workers = 0 # number of data loading workers
epochs = 300 #number of total epochs to run
patiences = 50 # number of epochs to tolerate no improvement of val_loss'
dp_rate = 0.2 # dropout rate


#get the test train data
train_data, test_data = get_train_test_data(test_id, data_cfg)

train_dataset = Hand_skeleton_Dataset(train_data, data_aug = True, time_len = 8)

test_dataset = Hand_skeleton_Dataset(test_data, data_aug = False, time_len = 8)


# dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = num_workers, pin_memory=False)

validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=True, num_workers = num_workers, pin_memory=False)

trained_model_path = "trained_weights//DHS_ID-{}_dp-{}_lr-{}_dc-{}/".format(test_id,dp_rate, learning_rate, data_cfg)
#os.mkdir(trained_model_path)

model = STA(class_num, dp_rate)
model = model.cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


criterion = torch.nn.CrossEntropyLoss()

train_data_num = 2660
test_data_num = 140
iter_per_epoch = int(train_data_num / batch_size)


# training parameter
max_acc = 0
no_improve_epoch = 0
n_iter = 0


########################### training and evaluations

for epoch in range(epochs):
    print("\ntraining.............")
    model.train()
    start_time = time.time()
    
    train_acc = 0
    train_loss = 0
    
    ############################## training
    
    for i, sample_batched in enumerate(train_loader):
        n_iter += 1
        
        if i + 1 > iter_per_epoch:
            continue
        
        data = sample_batched["skeleton"].float()
        data = data.cuda()
        score = model(data)
        
        label = sample_batched["label"]
        label = label.type(torch.LongTensor)
        label = label.cuda()
        label = torch.autograd.Variable(label, requires_grad=False)
        loss = criterion(score,label)
        
        score_ = score.cpu().data.numpy()
        label_ = label.cpu().data.numpy()
        output_ = np.argmax(score_, axis=1)
        accuracy = np.sum(output_ == label_)/float(label_.size)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_acc += accuracy
        train_loss += loss
        
    train_acc /= float(i + 1)
    train_loss /= float(i + 1)
    
    print("*** DHS  Epoch: [%2d] time: %4.4f, cls_loss: %.4f  train_ACC: %.6f ***" % (epoch + 1,  time.time() - start_time, train_loss.data, train_acc))
    
    start_time = time.time()
    
    
    ################################## evaluation
    
    with torch.no_grad():
        val_loss = 0
        acc_sum = 0
        
        model.eval()
        
        for i, sample_batched in enumerate(validation_loader):
            
            data = sample_batched["skeleton"].float()
            data = data.cuda()
            score = model(data)
            
            label = sample_batched["label"]
            label = label.type(torch.LongTensor)
            label = label.cuda()
            label = torch.autograd.Variable(label, requires_grad=False)
            loss = criterion(score,label)
            
            score_ = score.cpu().data.numpy()
            label_ = label.cpu().data.numpy()
            output_ = np.argmax(score_, axis=1)
            accuracy = np.sum(output_ == label_)/float(label_.size)
            
            val_loss += loss
            
            if i == 0:
                score_list = score
                label_list = label
            else:
                score_list = torch.cat((score_list, score), 0)
                label_list = torch.cat((label_list, label), 0)
                
            
        val_loss = val_loss / float(i + 1)
        
        score__ = score_list.cpu().data.numpy()
        label__ = label_list.cpu().data.numpy()
        output__ = np.argmax(score__, axis=1)
        val_cc = np.sum(output__ == label__)/float(label__.size)
        
        print("*** DHS  Epoch: [%2d], val_loss: %.6f, val_ACC: %.6f ***" % (epoch + 1, val_loss, val_cc))
        
        #################### save the best model
        
        if val_cc > max_acc:
            max_acc = val_cc
            no_improve_epoch = 0
            al_cc = round(val_cc, 10)
            
            torch.save(model.state_dict(), '{}/epoch_{}_acc_{}.pth'.format(trained_model_path, epoch + 1, val_cc))
            print("performance improve, saved the new model......best acc: {}".format(max_acc))
            
        else:
            no_improve_epoch += 1
            print("no_improve_epoch: {} best acc {}".format(no_improve_epoch,max_acc))
            
        if no_improve_epoch > patiences:
            print("stop training....")
            break
        
                
        
            
            
    
    
