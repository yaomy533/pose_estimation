#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/14 14:14
# @Author  : yaomy

# TODO: 在原有的dataset的基础上进行改进，一张图所有物体拼到一起，然后再改写colltefn
# TODO：使用iterableDataset

import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self):
        self.all_lists = list(range(0, 128))
        self.num = [2, 3, 4, 5]

    def __len__(self):
        return len(self.all_lists)

    def __getitem__(self, item):
        datas = {}
        torch.random.choice()
        img = torch.randn(3, 256, 256)
