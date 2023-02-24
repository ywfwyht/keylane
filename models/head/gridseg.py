'''
Date: 2023-01-18 05:36:39
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-02-22 06:18:34
FilePath: /K-Lane/home/work_dir/work/keylane/models/head/gridseg.py
'''
import torch
import torch.nn as nn
import numpy as np


class GridSeg(nn.Module):
    def  __init__(self, channel_1=1024, channel_2=2048, num_classes=7, cfg=None):
        super().__init__()

        self.cfg = cfg
        self.act_sigmoid = nn.Sigmoid()
        self.conf_pred = nn.Sequential(
            nn.Conv2d(channel_1, channel_2, 1),
            nn.Conv2d(channel_2, 1, 1)
        )
        self.cls_pred = nn.Sequential(
            nn.Conv2d(channel_1, channel_2, 1),
            nn.Conv2d(channel_2, num_classes, 1)
        )
    
    def forward(self, x):

        conf_out = self.act_sigmoid(self.conf_pred(x))
        class_out = self.cls_pred(x)

        out = torch.cat((class_out, conf_out), 1)

        # out = torch.argmax(class_out, axis=1) # for every class / onnx

        return out


if __name__=="__main__":
    head = GridSeg()
    input = torch.rand(1, 1024, 144, 144)
    out = head(input)
    print(out.shape)