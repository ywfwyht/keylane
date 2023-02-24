'''
Date: 2023-01-18 07:20:11
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-01-18 07:21:21
FilePath: /K-Lane/home/work_dir/work/keylane/models/loss/DiceLoss.py
'''
#Dice系数
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

#Dice损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5
    
    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score


if __name__=='__main__':

    loss = DiceLoss()
    predict = torch.randn(3, 4, 4)
    target = torch.randn(3, 4, 4)

    score = loss(predict, target)
    print(score)
