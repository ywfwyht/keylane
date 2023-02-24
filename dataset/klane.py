
'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os.path as osp
import os
from re import M
import numpy as np
import cv2
import torch
from glob import glob
import pickle
import open3d as o3d
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class KLane(Dataset):
    def __init__(self, data_root, split, mode_item='pc', description=None, cfg=None):
        self.cfg = cfg
        self.data_root = data_root
        self.mode_item = mode_item
        self.training = 'train' in split
        self.num_seq = len(os.listdir(osp.join(data_root, 'train')))
        self.list_data_type = ['bev_image', 'bev_image_label', 'bev_tensor', 'frontal_img', 'pc']
        self.list_data_tail = ['.pickle', '.pickle', '.pickle', '.jpg', '.pcd']
        
        if split == 'train':
            self.data_infos = self.load_train_data_infos()
        elif split == 'test':
            self.data_infos = self.load_test_data_infos()

        if description:
            self.data_infos = self.filter_data_infos(description)

    def get_time_string(self, data_file):
        return data_file.split('.')[0].split('_')[-1]


    def load_train_data_infos(self):
        data_infos = []
        train_root = osp.join(self.data_root, 'train')
        for name_seq in os.listdir(train_root):
            # criterion: tensor_label
            list_tensor_label = sorted(os.listdir(osp.join(train_root, name_seq, 'bev_tensor_label')))
            list_list_data = list(map(lambda data_type: os.listdir(osp.join(train_root, name_seq, data_type)), self.list_data_type))
            # print(list_list_data)
            temp_description = open(osp.join(train_root, name_seq, 'description.txt'), 'r')
            list_description = temp_description.readline()
            list_description = list_description.split(',')
            list_description[-1] = list_description[-1][:-1] # delete \n
            temp_description.close()

            for name_tensor_label in list_tensor_label:
                temp_data_info = dict()
                temp_data_info['bev_tensor_label'] = osp.join(train_root, name_seq, 'bev_tensor_label', name_tensor_label)
                temp_data_info['description'] = list_description
                time_string = self.get_time_string(name_tensor_label)
                
                for idx, data_type in enumerate(self.list_data_type):
                    temp_data_name = data_type + '_' + time_string + self.list_data_tail[idx]
                    if temp_data_name in list_list_data[idx]:
                        temp_data_info[data_type] = osp.join(train_root, name_seq, data_type, temp_data_name)
                    else:
                        temp_data_info[data_type] = None
                data_infos.append(temp_data_info)

        return data_infos

    def load_test_data_infos(self):
        data_infos = []
        train_root = osp.join(self.data_root, 'train')
        test_root = osp.join(self.data_root, 'test')
        
        test_descriptions_path = osp.join(self.data_root, 'description_frames_test.txt')
        list_test_descriptions = []
        with open(test_descriptions_path, 'r') as f:
            for line in f: # ['003552678455880', '5', 'night', 'highway', 'occ0']
                list_test_descriptions.append(line.strip('\n').split(', '))

        list_time_string = []
        list_corresponding_seq = []
        list_list_description = []
        list_list_data = [[],[],[],[],[]]
        # time_string, seq
        for name_seq in os.listdir(train_root):
            # criterion: 'bev_tensor'
            temp_list_time_string = list(map(self.get_time_string, sorted(os.listdir(osp.join(train_root, name_seq, 'bev_tensor')))))
            list_time_string.extend(temp_list_time_string)
            temp_list_corresponding_seq = [name_seq]*len(temp_list_time_string)
            list_corresponding_seq.extend(temp_list_corresponding_seq)
            
            temp_description = open(osp.join(train_root, name_seq, 'description.txt'), 'r')
            list_description = temp_description.readline()
            list_description = list_description.split(',')
            list_description[-1] = list_description[-1][:-1] # delete \n
            temp_description.close()
            list_list_description.append(list_description)
            
            temp_list_list_data = list(map(lambda data_type: os.listdir(osp.join(train_root, name_seq, data_type)), self.list_data_type))
            for i in range(len(temp_list_list_data)):
                list_list_data[i].extend(temp_list_list_data[i])
        # print(list_list_data)
        # print('list_test_descriptions:',list_test_descriptions)

        for name_tensor_label in sorted(os.listdir(test_root)):
            temp_data_info = dict()
            time_string = self.get_time_string(name_tensor_label)  # example: 005045434841760
            corresponding_idx = list_time_string.index(time_string)

            temp_data_info['bev_tensor_label'] = osp.join(test_root, name_tensor_label)
            name_seq = list_corresponding_seq[corresponding_idx]
            # temp_data_info['description'] = list_list_description[int(name_seq.split('_')[-1])-1]
            
            for desc in list_test_descriptions:
                if desc[0] == time_string:
                    temp_data_info['description'] = desc[-3:]
                    # if self.cfg.is_eval_conditional:
                    #     temp_data_info['description'] = desc
                    # else:
                    #     temp_data_info['description'] = desc[-3:] # modified by Xiaoxin
                    # break

            for idx, data_type in enumerate(self.list_data_type):
                temp_data_name = data_type + '_' + time_string + self.list_data_tail[idx]
                # print(temp_data_name)
                if temp_data_name in list_list_data[idx]:
                    temp_data_info[data_type] = osp.join(train_root, name_seq, data_type, temp_data_name)
                else:
                    temp_data_info[data_type] = None
            data_infos.append(temp_data_info)
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.mode_item == 'pillar':
            data_info = self.data_infos[idx]
            
            if not osp.isfile(data_info['bev_tensor']):
                raise FileNotFoundError('cannot find file: {}'.format(data_info['bev_tensor']))

            if not osp.isfile(data_info['bev_tensor_label']):
                raise FileNotFoundError('cannot find file: {}'.format(data_info['bev_tensor_label']))

            meta = data_info.copy()
            sample = dict()
            sample['meta'] = meta

            with open(meta['bev_tensor'], 'rb') as f:
                bev_tensor = pickle.load(f, encoding='latin1')
            with open(meta['bev_tensor_label'], 'rb') as f:
                bev_tensor_label = pickle.load(f, encoding='latin1')

            pillars = np.squeeze(bev_tensor[0], axis=0)
            pillar_indices = np.squeeze(bev_tensor[1], axis=0)

            sample['pillars'] = pillars
            sample['pillar_indices'] = pillar_indices
            sample['label'] = bev_tensor_label[:,0:144]
            sample['rowise_existence'] = bev_tensor_label[:,144:]

        elif self.mode_item == 'pc':
            data_info = self.data_infos[idx]

            if not osp.isfile(data_info['pc']):
                raise FileNotFoundError('cannot find file: {}'.format(data_info['pc']))

            if not osp.isfile(data_info['bev_tensor_label']):
                raise FileNotFoundError('cannot find file: {}'.format(data_info['bev_tensor_label']))

            meta = data_info.copy()
            sample = dict()
            sample['meta'] = meta
            # print(meta['bev_tensor_label'])
            # print(meta['pc'])

            with open(meta['bev_tensor_label'], 'rb') as f:
                bev_tensor_label = pickle.load(f, encoding='latin1')
            sample['label'] = bev_tensor_label[:,0:144]

            points = read_pcd(meta['pc'])
            data = point_projection(points)
            sample['proj'] = data['bev_img']

            sample['proj'] = np.transpose(sample['proj'], (2,0,1))
            sample['proj'] = sample['proj'].astype(np.float32)       

        return sample

def read_pcd(pcd_file, readline=11):

        ## np.fromfile read pcd 文本/二进制
        # cloud = np.fromfile(str(pcd_file), dtype=np.float32, count=-1)
        data = []
        with open(pcd_file, 'r') as f:
            lines = f.readlines()[readline:]           
            for line in lines:
                line = list(line.strip('\n').split(' '))
                x = float(line[0])
                y = float(line[1])
                z = float(line[2])
                i = int(line[3])
                r = float(line[4]) # for ouster lidar 
                data.append(np.array([x,y,z,i,r]))
            # points = list(map(lambda line: list(map(lambda x: float(x), line.split(' '))), lines))
            points = np.array(data)

        return points

def point_projection(points, list_roi_xyz = [0.02, 69.12, -11.52, 11.52, -2.0, 1.5],
                        list_grid_xy = [0.06, 0.02],
                        list_img_size_xy=[1152, 1152],
                        list_value_idx = [2, 3, 4],
                        list_list_range = [[-2.0, 1.5], [0, 128], [0, 32768]],
                        is_flip=False):

    x_min, x_max, y_min, y_max, z_min, z_max = list_roi_xyz
    x_grid, y_grid = list_grid_xy

    idx = np.where((points[:, 0] > x_min) & (points[:, 0] < x_max) &
                   (points[:, 1] > y_min) & (points[:, 1] < y_max) &
                   (points[:, 2] > z_min) & (points[:, 2] < z_max))
    points_roi = points[idx[0]]

    data = dict()
    data['points'] = points_roi

    list_xy_values = points_roi[:,:2].tolist()

    x_img = (points_roi[:,0] - x_min) // x_grid
    y_img = (points_roi[:,1] - y_min) // y_grid
    # x_img -= int(np.floor(x_min / x_grid))
    # y_img -= int(np.floor(y_min / y_grid))
    x_img = x_img.reshape(-1, 1).astype(int)
    y_img = y_img.reshape(-1, 1).astype(int)
    arr_xy_values = np.concatenate((x_img, y_img), axis=1)

    data.update({'img_idx': arr_xy_values})
    # data['img_idx'] = arr_xy_values

    n_channels = len(list_value_idx)
    temp_img = np.full((list_img_size_xy[0], list_img_size_xy[1], n_channels), 0, dtype=float)

    list_list_values = [] # z, intensity, reflectivity
    for channel_idx, value_idx in enumerate(list_value_idx):
        temp_arr = points_roi[:,value_idx].copy()

        # Normalize
        v_min, v_max = list_list_range[channel_idx]
        temp_arr[np.where(temp_arr<v_min)] = v_min
        temp_arr[np.where(temp_arr>v_max)] = v_max
        temp_arr = (temp_arr-v_min)/(v_max-v_min)
        # list_list_values.append(temp_arr)

        for idx, xy in enumerate(arr_xy_values):
            temp_img[xy[0], xy[1], channel_idx] = temp_arr[idx]

    if is_flip:
        temp_img = np.flip(np.flip(temp_img, 0), 1).copy()

    data['bev_img'] = temp_img # (1152, 1152, 3)

    return data

if __name__=='__main__':
    ### Checking None Data ###
    # checking_none_data(dataset_type='test')
    ### Checking None Data ###

    # visualize_pc_data_with_roi()

    from ..utils.config import Config
    data_path = '/home/data/klane'
    config_file = './configs/Proj28_GFC-T3_RowRef_82_73.py'
    cfg = Config.fromfile(config_file)
    dataset = KLane(data_path, 'train', cfg=cfg)
    for i in range(300):
        dataset.__getitem__(i)
    dataset1 = KLane(data_path, 'test', cfg=cfg)
    dataset1.__getitem__(1)



    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # print(len(loader))

    # from tqdm import tqdm
    # for batch_idx, batch_data in enumerate(loader):

    #     # print(batch_data)
        
    #     import cv2
    #     img = batch_data['proj'][0,1,:,:].detach().cpu().numpy()
    #     # img = np.transpose(img, (1, 2, 0))
    #     # img[np.where(img>0.5)] = 1
    #     print(img.shape)
    #     cv2.imwrite('./input_img.png', img*255)
    #     cv2.imwrite('./label.png', batch_data['label'][0].numpy().astype(np.uint8))
    #     # cv2.waitKey(0)

    #     import sys
    #     sys.exit()


