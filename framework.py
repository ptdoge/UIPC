import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import resnet
from ._blocks import Conv1x1, Conv3x3, get_norm_layer, ConvTransposed3x3
from ._utils import KaimingInitMixin, Identity
from kmeans_pytorch.kmeans_pytorch import kmeans
from ._blocks import Conv1x1, BasicConv
from torch import Tensor
import math
import numpy as np

from einops import rearrange
import os

class Backbone(nn.Module, KaimingInitMixin):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2,1,2,2,2)):
        super().__init__()

        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch, 
                64,
                kernel_size=7,
                stride=strides[0],
                padding=3,
                bias=False
            )
        
        if not pretrained:
            self._init_weight()

    def forward(self, x):
        # x 3 256 256
        x = self.resnet.conv1(x) # 64 128 128
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # 64 64 64

        x1 = self.resnet.layer1(x) # 64 64 64
        x2 = self.resnet.layer2(x1) # 128 32 32
        x3 = self.resnet.layer3(x2) # 256 16 16
        x4 = self.resnet.layer4(x3) # 512 8 8 

        return x1, x2, x3, x4

    def _trim_resnet(self):
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()

# Projector
class Decoder(nn.Module, KaimingInitMixin):
    def __init__(self, fc_ch):
        super().__init__()
        self.dr1 = Conv1x1(64, 96, norm=True, act=True)
        self.dr2 = Conv1x1(128, 96, norm=True, act=True)
        self.dr3 = Conv1x1(256, 96, norm=True, act=True)
        self.dr4 = Conv1x1(512, 96, norm=True, act=True)
      
        self.conv_out = nn.Sequential(
            Conv3x3(384, 256, norm=True, act=True),
            Conv1x1(256, fc_ch, norm=True, act=True)
        )

        self._init_weight()

    def forward(self, feats):

        f1 = self.dr1(feats[0]) # 96 64 64
        f2 = self.dr2(feats[1]) # 96 32 32
        f3 = self.dr3(feats[2]) # 96 16 16
        f4 = self.dr4(feats[3]) # 96 8 8

        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)
 
        x = torch.cat([f1, f2, f3, f4], dim=1) # 384 64 64

        x = self.conv_out(x)
        
        return x

# Predictor
class AuxDecoder(nn.Module, KaimingInitMixin):
    def __init__(self, fc_ch):
        super().__init__()
        self.dr1 = Conv1x1(64, 96, norm=True, act=True)
        self.dr2 = Conv1x1(128, 96, norm=True, act=True)
        self.dr3 = Conv1x1(256, 96, norm=True, act=True)
        self.dr4 = Conv1x1(512, 96, norm=True, act=True)
        self.conv_out = nn.Sequential(
            Conv3x3(384, 256, norm=True, act=True),
            Conv1x1(256, fc_ch, norm=True, act=True)
        )

        self._init_weight()

    def forward(self, feats):

        f1 = self.dr1(feats[0]) # 96 64 64
        f2 = self.dr2(feats[1]) # 96 32 32
        f3 = self.dr3(feats[2]) # 96 16 16
        f4 = self.dr4(feats[3]) # 96 8 8

        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)
 
        x = torch.cat([f1, f2, f3, f4], dim=1) # 384 64 64
        x = self.conv_out(x) # 64 64 64 

        return x

class Base(nn.Module):
    def __init__(self, in_ch, fc_ch=64):
        super().__init__()
        self.extract = Backbone(in_ch=in_ch, arch='resnet18') # 
        self.decoder = Decoder(fc_ch=fc_ch)
        self.crossc = CrossC(margin=2)

        self.max_epoch = 10 #  
        ## 
        out_ch = 1
        self.pseudo_label = AuxDecoder(fc_ch = fc_ch)
        self.class_layer = nn.Conv2d(fc_ch, out_ch, 3, 1, 1)


    def forward(self, t1, t2, epoch=1, split='test'):
        b, _, _, _ = t1.size()
        t = torch.cat([t1,t2],dim=0)
        feats = self.extract(t)

        f = self.decoder(feats)

        # ISSM
        pseudo_label = F.sigmoid(self.class_layer(self.pseudo_label(feats))) # 
        y1, y2 = torch.split(pseudo_label,b,dim=0)
        y = torch.abs(y1-y2)
        y = F.interpolate(y, size=t1.shape[2:], mode='bilinear', align_corners=True)
        
        if split == 'train' and epoch > self.max_epoch:
            region_contrast, _, _ = self.crossc(f, pseudo_label.detach()) # mask 64*64
        else:
            region_contrast = None

        f_1, f_2 =  torch.split(f,b,dim=0)
        
        dist = torch.norm(f_1 - f_2, dim=1, keepdim=True) # 1 64 64 distance
        
        dist = F.interpolate(dist, size=t1.shape[2:], mode='bilinear', align_corners=True)
 
        return dist, y, region_contrast

# UIPC
class CrossC(nn.Module):
    def __init__(self, margin=2):
        # 
        super().__init__()
        self.margin = margin
        self.ignore_label = -1

    def forward(self, x, mask):
        b, c, h, w = x.size() 

        mask = mask.view(-1,1,h*w)
        x = x.view(-1,c,h*w)

        cls0_indice = (mask<0.5).float().squeeze().sum(dim=-1) > 64 # 
        cls1_indice = (mask>0.5).float().squeeze().sum(dim=-1) > 64

        if cls0_indice.sum() > 1 and cls1_indice.sum() > 1:

            mask0 = (mask[cls0_indice]<0.5).float()
            x0 = x[cls0_indice]
            nonobject_centers = (x0*mask0).sum(dim=-1)/mask0.sum(dim=-1)

            mask1 = (mask[cls1_indice]>0.5).float()
            x1 = x[cls1_indice]
            object_centers = (x1*mask1).sum(dim=-1)/mask1.sum(dim=-1) #

            loss_contrast = torch.pow(torch.clamp(self.margin - torch.cdist(object_centers, nonobject_centers), min=0), 2).mean()
            
            return loss_contrast, object_centers, nonobject_centers
        else:
            return None, None, None


if __name__ == '__main__':
    model = NonLocalMetric(kernel_size=16, stride=16)
    t1 = torch.randn(8,64,64,64)
    t2 = torch.randn(8,64,64,64)
    dist = model(t1,t2)
    print(dist.size())