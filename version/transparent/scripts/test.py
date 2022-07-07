#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/27 15:46
# @Author  : yaomy
# 测试脚本，用完请删除

# --------------------------------------------------------
# 测试dataset
# --------------------------------------------------------
# import torch
# import time
# from torch.utils.data import Dataset, DataLoader
#
#
# class Ds(Dataset):
#     def __init__(self):
#         self.lsts = [i for i in range(100)]
#
#     def __len__(self):
#         time.sleep(0.1)
#         return len(self.lsts)
#
#     def __getitem__(self, item):
#         return self.lsts[item]
#
#
# def laod_data(iter):
#     return iter.__next__()
#
#
# dataset = Ds()
# dataloader = DataLoader(dataset, shuffle=True, num_workers=10)
#
# data_iter = dataloader.__iter__()
#
# for _ in range(10):
#     print(laod_data(data_iter))

# --------------------------------------------------------
# 计算内参
# --------------------------------------------------------
# import math
# # fx = (1920.0/2.0)/math.tan((69.40/2./180) * math.pi)
# # fy = (1080.0/2.0)/math.tan((42.56/2./180) * math.pi)
# fx = (1024.0/2.0)/math.tan((69.40/2./180) * math.pi)
# fy = (576.0/2.0)/math.tan((42.56/2./180) * math.pi)
# print(fx, fy)


# --------------------------------------------------------
# 测试 pykeops in GPU
# --------------------------------------------------------
# import pykeops
# pykeops.clean_pykeops()          # just in case old build files are still present
# pykeops.test_torch_bindings()    # perform the compilation


# import torch.nn.functional as F
# import torch
#
# t4d = torch.ones(8, 800)
# p1d = (100, 100)
#
# print(F.pad(t4d, p1d).shape)


# --------------------------------------------------------
# 测试 FasterRCNN torch 官网源码
# --------------------------------------------------------
# import torchvision
# import torch
#
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# images = torch.rand(4, 3, 600, 1200)
# boxes = torch.tensor([
#     [[100, 150, 200, 250], [100, 150, 200, 250], [100, 150, 200, 250], [100, 150, 200, 250]],
#     [[100, 150, 200, 250], [100, 150, 200, 250], [100, 150, 200, 250], [100, 150, 200, 250]],
#     [[100, 150, 200, 250], [100, 150, 200, 250], [100, 150, 200, 250], [100, 150, 200, 250]],
#     [[100, 150, 200, 250], [100, 150, 200, 250], [100, 150, 200, 250], [100, 150, 200, 250]],
# ])
# labels = torch.randint(1, 91, (4, 11))
# images = list(image for image in images)
#
# targets = []
# for i in range(len(images)):
#     d = dict()
#     d['boxes'] = boxes[i]
#     d['labels'] = labels[i]
#     targets.append(d)
#
# output = model(images, targets)
# model.eval()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)



# --------------------------------------------------------
# 测试 模型输出
# --------------------------------------------------------