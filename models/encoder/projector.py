'''
Date: 2023-01-06 08:43:05
Author: yang_haitao
LastEditors: yanghaitao yang_haitao@leapmotor.com
LastEditTime: 2023-02-07 06:18:36
FilePath: /K-Lane/home/work_dir/work/keylane/models/backbone/projector.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

class Projector(nn.Module):
    def __init__(self,
                resnet='resnet34',
                pretrained=False,
                replace_stride_with_dilation=[False, True, False],
                out_conv=True,
                in_channels=[64, 128, 256, -1],
                cfg=None):
        super(Projector, self).__init__()
        self.cfg = cfg
        self.resnet = ResNetWrapper(
            resnet=resnet,
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
            out_conv=out_conv,
            in_channels=in_channels,
            cfg=cfg
        )
    
    def forward(self, sample):

        out = self.resnet(sample)

        return out


def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def  conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                padding=dilation, groups=groups, bias=False, dilation=dilation)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print('pretrained model: ', model_urls[arch])
        state_dict = torch.load(model_urls[arch]) # 离线指定文件
        # state_dict = load_state_dict_from_url(model_urls[arch])
        model.load_state_dict(state_dict, strict=False)
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


class ResNetWrapper(nn.Module):
    def __init__(self,
                resnet='resnet34',
                pretrained=True,
                replace_stride_with_dilation=[False, False, False],
                out_conv=False,
                in_channels=[64, 128, 256, 512],
                cfg=None):
        super(ResNetWrapper, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.resnet = eval(resnet)(pretrained=pretrained,
                                replace_stride_with_dilation=replace_stride_with_dilation,
                                in_channels=in_channels)
        self.conv = None

        if out_conv:
            out_channel = 512
            for in_channel in reversed(self.in_channels):
                if in_channel < 0: continue
                out_channel = in_channel
                break
            self.conv = conv1x1(out_channel * self.resnet.expansion, 64)
    
    def forward(self, x):
        x = self.resnet(x)
        if self.conv:
            x = self.conv(x)
        
        return x


class ResidualBlockCBAM(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ResidualBlockCBAM, self).__init__()
        # channel attention 压缩H，W为1
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        
        # spatial attention kernel_size=7
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):

        max_out = self.sharedMLP(self.maxpool(x))
        avg_out = self.sharedMLP(self.avgpool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out = torch.max(x, dim=1, keepdim=True)[0]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                groups=1, width_per_group=64, replace_stride_with_dilation=None,
                norm_layer=None, in_channels=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be none '
                            'or a 3-element tuple, got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # bev image in_channel=3
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 此处第一层可加入CBAM
        self.cbam = ResidualBlockCBAM(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = in_channels
        # 第一个残差层不进行下采样
        self.layer1 = self._make_layer(block, in_channels[0], layers[0]) 
        self.layer2 = self._make_layer(block, in_channels[1], layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        if in_channels[2] > 0:
            self.layer3 = self._make_layer(block, in_channels[2], layers[2], stride=2,
                                            dilate=replace_stride_with_dilation[1])
        if in_channels[3] > 0:
            self.layer4 = self._make_layer(block, in_channels[3], layers[3], stride=2,
                                            dilate=replace_stride_with_dilation[2])
        # 最后一个layer后添加CBAM
        self.expansion = block.expansion
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.weight, 0)
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                self._norm_layer(planes*block.expansion)
            )
        layers = []
        # 第一个残差块
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self._norm_layer))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=self._norm_layer))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.cbam(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.in_channels[2] > 0:
            x = self.layer3(x)
        if self.in_channels[3] > 0:
            x = self.layer4(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1 # 通道升降维倍数
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                    base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # 第一个卷积，通过stride进行下采样
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        # 第二个卷积不进行下采样
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        # 可加入CBAM

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 此处加入CBAM

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out



if __name__=="__main__":
    proj = Projector(resnet='resnet34',
                    pretrained=False,
                    replace_stride_with_dilation=[False, True, False],
                    out_conv=True,
                    in_channels=[64, 128, 256, -1],)

    print(proj)

    # cbam = ResidualBlockCBAM(32, reduction=6)
    # print(cbam)
    # input = torch.randn(1, 32, 4, 3)
    # out = cbam(input)
    # print('fffffff ', out.shape, out[:,1,1,1])

