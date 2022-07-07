#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/14 16:03
# @Author  : yaomy
import torch
import numpy as np
import random
import torch.backends.cudnn

seed = 1024
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
