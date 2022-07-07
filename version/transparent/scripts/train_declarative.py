#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 22:38
# @Author  : yaomy

# TODO: 将pvn3d的最小二乘拟合搬过来，同时写好其梯度的反向传播
# TODO: 将这个最小二乘拟合加入我们网络的后面，其对应关系为knn选取的对应关系
import sys
sys.path.append('/root/Workspace/project/transparent')
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from datasets.cleargrasp.dataset import PoseDataset
from lib.networks.network import PoseNet
from lib.networks.loss import PossLoss
from lib.declarative.leastsquares import LeastSquaresLayer
from pykeops.torch import generic_argkmin

dataset_root = '/root/Source/ymy_dataset/cleargrasp'
noise_trans = 0.03
# 预测多少组位姿
num_points = 1000


num_objects = 5
nepoch = 100
lr = 1e-4
w = 0.02
dataset = PoseDataset('train', 1000, True, dataset_root, noise_trans, False)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')

estimator = PoseNet(num_points=num_points, num_obj=num_objects)
estimator.cuda()
estimator.train()

optmizer = optim.Adam(estimator.parameters(), lr=lr)
diff_fit = LeastSquaresLayer()

for epoch in range(nepoch):
    np.random.seed()
    for i, datas in enumerate(dataloader, 0):

        if not datas['flag']:
            continue

        for index in range(len(datas['img_cropeds'])):
            optmizer.zero_grad()
            pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask, pred_boundry, choose = estimator(
                datas['img_cropeds'][index].cuda(),
                datas['intrinsic'][index].cuda(),
                datas['xmaps'][index].cuda(),
                datas['ymaps'][index].cuda(),
                datas['d_scales'][index].cuda(),
                datas['obj_ids'][index].cuda(),
            )
            # pred_r [bs, 1000, 4]
            # pred_t [bs*1000, 3]
            # pred_c [bs*1000, 1]

            # model_points [bs*1000, 1000, 3]
            # pred_matrix [bs*1000, 3, 3]
            # pred_points [bs*1000, 1000, 3]
            # model_points [bs*1000, 1000, 3]
            # target [bs*1000, 1000, 3]

            bs = pred_r.size(0)
            pred_c = pred_c.view(bs*num_points)
            model_points = datas['model_points'][index]
            num_cloud = model_points.size(1)
            model_points = model_points.view(bs*num_points, 1, 3).repeat(1, num_cloud, 1).cuda()
            final_r = PossLoss.predr2rotation(pred_r, pred_r.size(0), num_points)
            final_t = pred_t.view(bs*num_points, 3)
            target = datas['targets'][index].cuda()
            # pred_points = model_points @ final_r.permute(0, 2, 1) + final_t.view(bs * num_points, 1, 3).cuda()
            loss_list = []
            print('\n*******Start Optmize*******')
            for step in range(10):
                pred_points = model_points @ final_r.permute(0, 2, 1) + final_t.view(bs * num_points, 1, 3).cuda()
                inds = knn(pred_points.view(-1, 3).detach(), target[0].contiguous().detach())
                # print(inds.shape)
                target_corr = torch.index_select(target, 1, inds.view(-1))

                opt_R, opt_t = diff_fit(pred_points, target_corr.view(bs*num_points, num_cloud, 3))
                # opt_points = pred_points @ opt_R.permute(0, 2, 1) + opt_t.view(bs*num_points, 1, 3)

                final_points = model_points @ opt_R.permute(0, 2, 1) + opt_t.view(bs * num_points, 1, 3).cuda()
                dis = torch.mean(torch.norm((final_points - model_points), dim=2), dim=1)
                loss = torch.mean((dis * pred_c - w * torch.log(pred_c + 1e-8)), dim=0)
                loss_list.append(loss)
                # loss.backward(retain_graph=True)
                final_r_grad = torch.autograd.grad(loss, final_r, retain_graph=True)
                pred_r_grad = torch.autograd.grad(loss, pred_r, retain_graph=True)
                # print(final_r_grad)
                # print(pred_r_grad)

                final_r = final_r @ opt_R
                final_t = final_t + opt_t
                print(f'step {step} ', loss)
            # sum(loss_list).backward(retain_graph=True)
            # r_grad = torch.autograd.grad(loss_list[0], pred_r)
            # print(r_grad)
            # t_grad = torch.autograd.grad(sum(loss_list), pred_t)
            # c_grad = torch.autograd.grad(sum(loss_list), pred_c)
            # print(estimator.pose_predictor.conv1_r.weight.grad)
            # exit()
            optmizer.step()



