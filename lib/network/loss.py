#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 10:13
# @Author  : yaomy
import os
import torch
import math
import torch.nn as nn
from pathlib import Path
from mmcv import Config
from lib.network.loss_utils import MapLoss, l1, cosine, cross_entropy, get_xyz

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETCT_PATH = Path(os.path.realpath(__file__)).parent.parent.parent
CONFIG = Config.fromfile(f'{PROJETCT_PATH}/config/linemod/lm_v3.py')


class PoseLoss(nn.Module):
    def __init__(self, sym_list, knn, num_point=1000):
        super(PoseLoss, self).__init__()
        self.num_point = num_point
        self.sym_list = sym_list
        self.knn = knn

    def forward(self, pred_r, pred_t, targets, model_points, idxs):
        pred_points = model_points @ pred_r.permute(0, 2, 1) + pred_t  # [bs, N, 3]
        bs = pred_points.size(0)
        tgts = list()
        for b in range(bs):
            idx = idxs[b]
            tgt = targets[b]
            pred_point = pred_points[b]
            if idx in self.sym_list:
                inds = self.knn(pred_point, tgt)
                # print(inds.shape, inds.max(), inds.min(), tgt.shape)
                tgt = torch.index_select(tgt, 0, inds.view(-1))

            tgts.append(tgt)
        targets = torch.stack(tgts, dim=0)
        dis = torch.mean(torch.norm((pred_points - targets), dim=2), dim=1).mean()
        return dis


class KRRNLoss(nn.Module):
    def __init__(self, sym_list, knn, cfg=CONFIG):
        super(KRRNLoss, self).__init__()
        self.cfg = cfg
        self.XYZLoss = MapLoss(fn=l1, reduction='elementwise_mean')
        self.NLoss = MapLoss(fn=cosine, reduction='elementwise_mean')
        self.RegionLoss = MapLoss(fn=cross_entropy, reduction='elementwise_mean')
        # self.MaskLoss = nn.L1Loss(reduction="mean")
        # 多分类损失
        self.MaskLoss = MapLoss(fn=cross_entropy, reduction='elementwise_mean')

        self.PoseLoss = PoseLoss(sym_list, knn)
        self.loss_weight = self.cfg.Train.Loss.LOSS_WEIGHT

    def forward(self, pred, gt, opt_pose=False):
        # loss_xyz = self.XYZLoss(pred['xyz'], gt['xyz'])
        # pred_xyz = get_xyz(pred['xyz'], pred['region'], gt['region_point'])
        pred_xyz = pred['xyz']
        loss_xyz = self.XYZLoss(pred_xyz, gt['xyz'])

        loss_normal = self.NLoss(pred['normal'], gt['normal'])
        loss_region = self.RegionLoss(pred['region'], gt['region'].unsqueeze(1).long())
        loss_mask = self.MaskLoss(pred['mask'], gt['multi_cls_mask'].unsqueeze(1).long())
        if opt_pose:
            loss_add = self.PoseLoss(gt['target_r'], pred['pred_t'].unsqueeze(1), gt['target'], gt['model_points'],
                                     gt['cls_id'])
        else:
            loss_add = 0
        loss = self.loss_weight['weight_xyz'] * loss_xyz \
               + self.loss_weight['weight_region'] * loss_region \
               + self.loss_weight['weight_mask'] * loss_mask \
               + self.loss_weight['weight_normal'] * loss_normal \
               + self.loss_weight['weight_pose'] * loss_add

        return {
            'loss': loss,
            'loss_add': loss_add,
            'loss_xyz': loss_xyz,
            'loss_region': loss_region,
            'loss_normal': loss_normal,
            'loss_mask': loss_mask,
        }
