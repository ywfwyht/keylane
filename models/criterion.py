'''
Date: 2023-02-14 01:52:49
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-02-20 02:11:44
FilePath: /K-Lane/home/work_dir/work/keylane/models/criterion.py
'''
import torch
import torch.nn as nn
import numpy as np

from models.loss.DiceLoss import DiceLoss


def loss_criterion(pred, label):
    label = label[:,:, :144]

    # Output image: top-left of the image is farthest-left
    num_of_labels = len(label)
    label_tensor = np.zeros((num_of_labels, 2, 144, 144), dtype = np.longlong)

    for k in range(num_of_labels):
        label_temp = np.zeros((144,144,2), dtype = np.longlong)
        label_data = label[k]

        for i in range(144):
            for j in range(144):
                y_idx = 144 - i - 1
                x_idx = 144 - j - 1
                # y_idx = i
                # x_idx = j

                line_num = int(label_data[i][j])
                if line_num == 255:
                    label_temp[y_idx][x_idx][1] = 0
                    # classification
                    label_temp[y_idx][x_idx][0] = 6
                else: # 클래스
                    # confidence
                    label_temp[y_idx][x_idx][1] = 1
                    # classification
                    label_temp[y_idx][x_idx][0] = line_num

        label_tensor[k,:,:,:] = np.transpose(label_temp, (2, 0, 1))

    lanes_label = torch.tensor(label_tensor)

    pred_cls = pred[:, 0:7, :, :]
    pred_conf = pred[:, 7, :, :]

    label_cls = lanes_label[:, 0, :, :].cuda()
    label_conf = lanes_label[:, 1, :, :].cuda()

    cls_loss = 0.
    cls_loss += nn.CrossEntropyLoss()(pred_cls, label_cls)

    diceloss = DiceLoss()
    conf_loss = diceloss(pred_conf, label_conf)

    loss = cls_loss + conf_loss

    loss_dict = {'loss':loss, 'cls_loss':cls_loss, 'conf_loss':conf_loss}

    return loss_dict

