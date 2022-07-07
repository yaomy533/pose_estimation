#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/3 17:33
# @Author  : yaomy

from datasets.cleargrasp.dataset import BathPoseDataset
import torch
from pathlib import Path


class Dataset(BathPoseDataset):
    def __init__(self):
        super(Dataset, self).__init__('train', 1000, True, '/root/Source/ymy_dataset/cleargrasp', 0., False)

    def __getitem__(self, index):
        cc = torch.tensor([0.])
        path = [self._datalist_input[index]]
        return cc, path


dataset = Dataset()
_, p = dataset.__getitem__(0)
print(p)

