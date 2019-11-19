# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 03:22:30 2019

@author: kb
"""

import random
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from random import randint,shuffle

class Hand_skeleton_Dataset(Dataset):
    
    def __init__(self, data, time_len, data_aug):
        
        # here data = {video, lable}
        # do_data_aug = data augmentation
        
        self.data_aug = data_aug
        self.data = data
        self.time_len = time_len
        self.compoent_num = 22
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        data_element = self.data[idx]
        
        hand_skeleton = data_element['skeleton']
        hand_skeleton = np.array(hand_skeleton)
        
        if self.data_aug:
            hand_skeleton = self.do_data_aug(hand_skeleton)
            
        # sample frames
        data_num = hand_skeleton.shape[0]
        idx_list = self.get_sample_frame(data_num)
        
        hand_skeleton = [hand_skeleton[idx] for idx in idx_list]
        hand_skeleton = np.array(hand_skeleton)
        
        # norm by center
        hand_skeleton -= hand_skeleton[0][1]
        
        hand_skeleton_tensor = torch.from_numpy(hand_skeleton).float()
        
        label = data_element["label"] - 1
        
        sample = {'skeleton': hand_skeleton_tensor, "label" : label}
        
        return sample
    
    def do_data_aug(self, hand_skeleton):
        
        # scale
        def scaling(hand_skeleton):
            ratio = 0.2
            factor = np.random.uniform(1 - ratio, 1 + ratio)
            video_len = hand_skeleton.shape[0]
            
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    hand_skeleton[t][j_id] *= factor
                    
            hand_skeleton = np.array(hand_skeleton)
            return hand_skeleton
        
        
        def shifting(hand_skeleton):
            low = -0.1
            high = -low
            offset = np.random.uniform(low, high, 3)
            video_len = hand_skeleton.shape[0]
            for t in range(video_len):
                for j_id in range(self.compoent_num):
                    hand_skeleton[t][j_id] += offset
                    
            hand_skeleton = np.array(hand_skeleton)
            return hand_skeleton
        

        def noise_addition(hand_skeleton):
            low = -0.1
            high = -low
            #select 4 joints
            all_joint = list(range(self.compoent_num))
            shuffle(all_joint)
            selected_joint = all_joint[0:4]

            for j_id in selected_joint:
                noise_offset = np.random.uniform(low, high, 3)
                for t in range(self.time_len):
                    hand_skeleton[t][j_id] += noise_offset
                    
            hand_skeleton = np.array(hand_skeleton)
            return hand_skeleton
        
        
        def time_interpolate(hand_skeleton):
            hand_skeleton = np.array(hand_skeleton)
            video_len = hand_skeleton.shape[0]

            r = np.random.uniform(0, 1)

            result = []

            for i in range(1, video_len):
                displace = hand_skeleton[i] - hand_skeleton[i - 1]
                displace *= r
                result.append(hand_skeleton[i -1] + displace)# r*disp

            while len(result) < self.time_len:
                result.append(result[-1]) #padding
            result = np.array(result)
            return result
        
        
        # do aug
        aug_id = randint(0, 3)
        
        if aug_id == 0:
            hand_skeleton = scaling(hand_skeleton)
        
        elif aug_id == 1:
            hand_skeleton = shifting(hand_skeleton)
        
        elif aug_id == 2:
            hand_skeleton = noise_addition(hand_skeleton)
            
        elif aug_id == 3:
            hand_skeleton = time_interpolate(hand_skeleton)
            
        return hand_skeleton
    
    
    def get_sample_frame(self, data_num):
        
        sample_size = self.time_len
        each_num = (data_num - 1) / (sample_size - 1)
        idx_list = [0, data_num - 1]
        
        for i in range(sample_size):
            index = round(each_num * i)
            
            if index not in idx_list and index < data_num:
                idx_list.append(index)
            
        idx_list.sort()
        
        while len(idx_list) < sample_size:
            idx = random.randint(0, data_num - 1)
            
            if idx not in idx_list:
                idx_list.append(idx)
                
        idx_list.sort()
        
        return idx_list
    
    
    
    
            
        
    



















            