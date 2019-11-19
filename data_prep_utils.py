# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 02:47:11 2019

@author: kb
"""

data_root_path = "DHG2016"

def data_parse(file_path):
    video = []
    
    for line in file_path:
        line = line.split('\n')[0]
        data = line.split(' ')
        
        frames = []
        points = []
        
        for data_ in data:
            points.append(float(data_))
            
            if len(points) == 3:
                frames.append(points)
                points = []
                
        video.append(frames)
    return video

def read_data():
    
    dataset = {}
    
    for gid in range(1, 15):
        print("gesture {} / {}".format(gid,14))
        for fid in range(1, 3):
            for subid in range(1, 21):
                for eid in range(1, 6):
                    
                    file_path = open(data_root_path + '/gesture_{}/finger_{}/subject_{}/essai_{}/skeleton_world.txt'.format(gid, fid, subid, eid))
                    video = data_parse(file_path)
                    
                    key = '{}_{}_{}_{}'.format(gid, fid, subid, eid)
                    dataset[key] = video
                    
                    file_path.close()
                    
    return dataset



def valid_frame(video):
    
    info_file = open(data_root_path + '/informations_troncage_sequences.txt')
    
    used_key = []
    
    for line in info_file:
        line = line.split('\n')[0]
        data = line.split(' ')
        
        gid = data[0]
        fid = data[1]
        subid = data[2]
        eid = data[3]
        
        key = "{}_{}_{}_{}".format(gid, fid, subid, eid)
        used_key.append(key)
        start_frame = int(data[4])
        end_frame = int(data[5])
        data = video[key]
        video[key] = data[(start_frame): end_frame + 1]
        
    return video


def train_test_split(test_id, filtered_video, cfg):
    # cfg = 0(14)       cfg = 1(28)
    
    train_data = []
    test_data = []
    
    for gid in range(1, 15):
        print("gesture {} / {}".format(gid,14))
        for fid in range(1, 3):
            for subid in range(1, 21):
                for eid in range(1, 6):
                    
                    key = "{}_{}_{}_{}".format(gid, fid, subid, eid)
                    
                    if cfg == 0:
                        label = gid
                    elif cfg == 1:
                        if fid == 1:
                            label = gid
                        else:
                            label = gid + 14
                            
                    data = filtered_video[key]
                    sample = {"skeleton":data, "label":label}
                    
                    if subid == test_id:
                        test_data.append(sample)
                    else:
                        train_data.append(sample)
                        
                        
    if len(test_data) == 0:
        raise "no such test subject"
        
    return train_data, test_data

def get_train_test_data(test_id, cfg):
    
    print('reading data from ' + data_root_path)
    video_data = read_data()
    
    print("filtering frames .......")
    filtered_video_data = valid_frame(video_data)
    train_data, test_data = train_test_split(test_id,filtered_video_data,cfg)
    
    return train_data,test_data