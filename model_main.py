# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:39:00 2019

@author: kb
"""

from model_utils import *
import torch.nn as nn
import torch

class STA(nn.Module):
    
    def __init__(self, num_classes, dp_rate):
        super(STA, self).__init__()
        
        h_dim = 32
        h_num= 8
        
        self.input_map = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), NormLayer(128), nn.Dropout(dp_rate))
        
        self.spatial_att = ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, att_type="spatial", time_len = 8)
        
        self.temporal_att = ATT_Layer(input_size=128,output_size= 128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate, att_type="temporal", time_len = 8)
        
        self.classes = nn.Linear(128, num_classes)
        
        
    def forward(self, x):
        
        time_len = x.shape[1]
        joint_num = x.shape[2]
        
        x = x.reshape(-1, time_len * joint_num,3)
        
        x = self.input_map(x)
        
        x = self.spatial_att(x)
        
        x = self.temporal_att(x)
        
        x = x.sum(1) / x.shape[1]
        
        pred = self.classes(x)
        
        return pred
    
    