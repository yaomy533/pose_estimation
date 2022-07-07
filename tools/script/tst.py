#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/15 16:41
# @Author  : yaomy

import torch

data = torch.randn(8, 15, 3, 64, 64)
idx = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8]).view(8, 1, 1, 1, 1).repeat(1, 1, 3, 64, 64)

print(torch.gather(data, 1, idx).shape)
