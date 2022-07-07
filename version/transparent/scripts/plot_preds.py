#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 13:57
# @Author  : yaomy
import torch
import numpy as np
from lib.viz.visualization import DrawPred
from lib.transform.rotation import quaternion_matrix_torch


class DrawCGPred(DrawPred):
    def __init__(self, model, dataloader):
        super(DrawCGPred, self).__init__(model, dataloader)
        self.num_points = 1000
        self.num_obj = 5
        self.bs = 1

    def get_pred(self, data):
        with torch.no_grad():
            pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask = self.model(
                data['img_cropeds'].cuda(),
                data['intrinsic'].cuda(),
                data['bboxes'].cuda(),
                data['d_scales'].cuda(),
                data['obj_ids'].cuda()
            )
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)
        pred_c = pred_c.view(self.bs, self.num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
        R = pred_r[0][which_max[0]].view(-1)
        T = pred_t[which_max[0]].type(torch.float)
        R = quaternion_matrix_torch(R).type(torch.float)[:3, :3]
        return R, T

    def load_data(self):
        while True:
            datas = super(DrawCGPred, self).load_data()
            if not datas['flag'].item():
                continue
            else:
                break
        index = np.random.randint(0, len(datas['img_cropeds']))
        data = {k: v[int(index)] for k, v in datas.items() if len(v) == len(datas['img_cropeds'])}
        return data
