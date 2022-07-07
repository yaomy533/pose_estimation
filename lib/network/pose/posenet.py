#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 21:39
# @Author  : yaomy

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from pathlib import Path
from mmcv import Config

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETCT_PATH = Path(os.path.realpath(__file__)).parent.parent.parent.parent
CONFIG = Config.fromfile(f'{PROJETCT_PATH}/config/linemod/lm_v3.py')


class RotBase(nn.Module):
    def __init__(self, cfg=CONFIG):
        super(RotBase, self).__init__()
        self.f = cfg.Module.POSENet.INC_R
        self.k = cfg.Module.POSENet.OUTC_R

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)

        self.ap1 = torch.nn.AvgPool1d(cfg.Data.NUM_POINTS)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        x = self.ap1(x)

        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.drop1(x)
        x = self.conv4(x)
        x = x.squeeze(2)
        x = x.contiguous()

        return x


class TBase(nn.Module):
    def __init__(self, cfg=CONFIG):
        super(TBase, self).__init__()
        self.f = cfg.Module.POSENet.INC_R + cfg.Module.NUM_CLS
        self.k = cfg.Module.POSENet.OUT_T
        # self.num_cls = cfg.Module.NUM_CLS

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)

        # x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        return xt


class PoseNet(nn.Module):
    def __init__(self, cfg=CONFIG):
        super(PoseNet, self).__init__()
        # self.rot_green = RotBase(cfg)
        # self.rot_red = RotBase(cfg)
        self.t_net = TBase(cfg)

    def forward(self, feat):
        # rc1 = self.rot_green(feat)
        # rc2 = self.rot_red(feat)
        rc1, rc2 = None, None
        t = self.t_net(feat)
        return rc1, rc2, t


def main():
    model = TBase()
    tf = torch.randn(8, 1664, 1024)
    m = nn.AvgPool1d(3, stride=2)
    for name, param in m.named_parameters():
        if param.requires_grad:
            print(name)

if __name__ == '__main__':
    main()
