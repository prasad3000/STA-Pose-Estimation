# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 07:32:34 2019

@author: kb
"""

import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np

class PEFunction(nn.Module):
    
    def __init__(self, ft_size, time_len, att_type):
        super(PEFunction, self).__init__()
        
        self.joint_num = 22
        self.time_len = time_len
        self.att_type = att_type
        
        if att_type =='spatial':
            
            # spatial position embedding            
            pos_list = []
            
            for t in range(time_len):
                for j_id in range(self.joint_num):
                    pos_list.append(j_id)
                    
        
        if att_type == 'temporal':
            
            # temporal position embedding
            pos_list = list(range(self.joint_num * self.time_len))
            
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        
        # in log space
        PE = torch.zeros(self.time_len * self.joint_num, ft_size)
        div_term = torch.exp(torch.arange(0, ft_size, 2).float() * -(math.log(10000.0) / ft_size))
        
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        PE = PE.unsqueeze(0).cpu()
        self.register_buffer('PE', PE)
    
    
    def forward(self, x):
        x = x + self.PE[:, :x.size(1)]
        return x


class NormLayer(nn.Module):
    
    "normalize the input vector"
    def __init__(self, ft_dim, eps=1e-6):
        super(NormLayer, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ft_dim))
        self.b_2 = nn.Parameter(torch.zeros(ft_dim))
        self.eps = eps

    def forward(self, x):
        #[batch, time, ft_dim)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class MultiHeadedAttention(nn.Module):
    
    def __init__(self, h_num, h_dim, input_dim, dp_rate,att_type):
        super(MultiHeadedAttention, self).__init__()
        
        self.h_dim = h_dim # head dimension
        self.h_num = h_num #head num
        self.attn = None #calculate att weight
        self.att_type = att_type  # spatial of  tempoal
        
        self.register_buffer('tempoal_mask', self.get_att_type_mask()[0])
        self.register_buffer('spatial_mask', self.get_att_type_mask()[1])
        
        # key query value maping
        
        self.key_map = nn.Sequential(nn.Linear(input_dim, self.h_dim * self.h_num), nn.Dropout(dp_rate))


        self.query_map = nn.Sequential(nn.Linear(input_dim, self.h_dim * self.h_num), nn.Dropout(dp_rate))


        self.value_map = nn.Sequential(nn.Linear(input_dim, self.h_dim * self.h_num), nn.ReLU(), nn.Dropout(dp_rate))
        
    
    def get_att_type_mask(self):
        
        #3.4
        time_len = 8
        joint_num = 22
        
        tempoal_mask = torch.ones(time_len * joint_num, time_len * joint_num)
        filted_area = torch.zeros(joint_num, joint_num)
        
        for i in range(time_len):            
            tempoal_mask[i * joint_num: i * joint_num + joint_num, i * joint_num: i * joint_num + joint_num] *= filted_area
            
        I = torch.eye(time_len * joint_num)
        spatial_mask = Variable((1 - tempoal_mask)).cpu()
        tempoal_mask = Variable(tempoal_mask + I).cpu()
        return tempoal_mask, spatial_mask
    
    
    def compute_attention(self,query, key, value):
        
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if self.att_type is not None:
            
            if self.att_type == "temporal":
                scores *= self.tempoal_mask
                scores += (1 - self.tempoal_mask) * (-9e15)
                
            elif self.att_type == "spatial":
                scores *= self.spatial_mask
                scores += (1 - self.spatial_mask) * (-9e15)
                
        att_sm = F.softmax(scores, dim=-1)
        
        return torch.matmul(att_sm, value), att_sm
    
    
    def forward(self, x):
        
        # x = [batch, t, dim]
        batch = x.size(0)
        
        '''
        1. linear projection
        2. attn
        3. concat
        '''
        
        query = self.query_map(x).view(batch, -1, self.h_num, self.h_dim).transpose(1, 2)
        key = self.key_map(x).view(batch, -1, self.h_num, self.h_dim).transpose(1, 2)
        value = self.value_map(x).view(batch, -1, self.h_num, self.h_dim).transpose(1, 2)
        
        x, self.attn = self.compute_attention(query, key, value)
        
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h_dim * self.h_num)
        
        return x
    

class ATT_Layer(nn.Module):
    
    def __init__(self, input_size, output_size, h_num, h_dim, dp_rate, time_len, att_type):
        super(ATT_Layer, self).__init__()
        
        self.PE = PEFunction(input_size, time_len, att_type)
        
        self.attn = MultiHeadedAttention(h_num, h_dim, input_size, dp_rate, att_type)
        
        self.ft_map = nn.Sequential(nn.Linear(h_num * h_dim, output_size), nn.ReLU(), NormLayer(output_size), nn.Dropout(dp_rate))
        
        self.init_parameters()
        
    def forward(self, x):
        
        x = self.PE(x)
        x = self.attn(x)
        x = self.ft_map(x)
        return x
    
    def init_parameters(self):
        model_list = [ self.attn, self.ft_map]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform(p)
    
    
    
    
    
        
    
            
        