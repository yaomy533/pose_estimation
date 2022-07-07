#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 13:40
# @Author  : yaomy

import torch
import torch.utils.data
from mmcv import Config
from torch.autograd import Variable

from lib.networks.TRPES import TRPESNet
from datasets.cleargrasp.dataset import BathPoseDataset as BathPoseDataset_trans
from lib.networks.loss import Loss, MultiLoss


dataset_root = '/root/Source/ymy_dataset/cleargrasp'
config_file = 'config/cleargrasp_config.py'
resume_posenet = '/root/Source/ymy_dataset/trained/instance/cleargrasp/26/pose_model_43_0.014258730074056402.pth'
w = 0.015
# 读取参数
cfg = Config.fromfile(config_file)
loss_weight = cfg.TRAIN.LOSS_WEIGHT
test_dataset = BathPoseDataset_trans('test', 1000, False, dataset_root, 0.0, True)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
sym_list = test_dataset.get_sym_list()
num_points_mesh = test_dataset.get_num_points_mesh()


estimator = TRPESNet(num_points=1000, num_obj=5)
estimator.cuda()
criterion = MultiLoss(1000, sym_list, loss_weight=loss_weight)
criterion.cuda()

if resume_posenet:
    # ckpt = torch.load(resume_posenet)
    estimator.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(resume_posenet).items()})

data_iter = testdataloader.__iter__()
while True:
    datas = next(data_iter)
    pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask = estimator(
        Variable(datas['img_cropeds']).cuda(),
        Variable(datas['intrinsic']).cuda(),
        Variable(datas['xmaps']).cuda(),
        Variable(datas['ymaps']).cuda(),
        Variable(datas['d_scales']).cuda(),
        Variable(datas['obj_ids']).cuda(),
    )
    print(pred_normal.shape)
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(pred_normal.detach().squeeze().permute(1, 2, 0).cpu().numpy())
    plt.subplot(122)
    plt.imshow(datas['normals'].detach().squeeze().permute(1, 2, 0).cpu().numpy())
    plt.show()

    loss, loss_dict = criterion(
        pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask,
        targets=datas['targets'].cuda(),
        model_points=datas['model_points'].cuda(),
        idxs=datas['obj_ids'].cuda(),
        w=w,
        gt_ns=datas['normals'].cuda(),
        gt_ds=datas['depths'].cuda(),
        gt_ms=datas['masks'].cuda(),
        axises=datas['axis'].cuda(),
        gt_rs=datas['target_rs'].cuda(),
    )
    break
