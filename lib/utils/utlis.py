#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/18 16:39
# @Author  : yaomy
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict, Counter


def batch_intrinsic_transform(K):
    """ K [bs, 4] ->  [bs, 3, 3]
        K [bs, 3, 3] ->  [bs, 4]
    """
    bs = K.size(0)
    device = K.device
    if K.dim() == 2 and K.size(-1) == 4:
        fxs, fys, cxs, cys = K[:, 0], K[:, 1], K[:, 2], K[:, 3]
        K_matrix_list = []
        for i in range(bs):
            fx, fy, cx, cy = fxs[i].view(-1), fys[i].view(-1), cxs[i].view(-1), cys[i].view(-1)
            K_matrix = torch.stack([
                torch.cat([fx, torch.tensor([0.]).to(device), cx]),
                torch.cat([torch.tensor([0.]).to(device), fy, cy]),
                torch.tensor([0., 0., 1.]).to(device),
            ], dim=0)
            K_matrix_list.append(K_matrix)
        return torch.stack(K_matrix_list)

    elif K.dim() == 3 and K.size(-1) == 3:
        return torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], dim=1)
    else:
        raise Exception('Wrong Shape!')


def load_part_module(modle: nn.Module, cpkt, distributed=False, backbone_only=False):
    cpkt_weight = torch.load(cpkt, map_location=torch.device('cpu'))

    modle_dict = modle.state_dict()
    if distributed:
        modle_dict = OrderedDict({k.replace('module.', ''): v for k, v in cpkt_weight.items()})

    if backbone_only:
        state_dict = OrderedDict({k: v for k, v in cpkt_weight.items() if (k in modle_dict.keys() and 'backbone' in k)})
    else:
        state_dict = OrderedDict({k: v for k, v in cpkt_weight.items() if (k in modle_dict.keys())})


    modle_dict.update(state_dict)
    modle.load_state_dict(modle_dict)
    return modle


def choose_ransac_batch(coor_emb, uv_emb, K_matrix, n=125):
    ncs = []
    bs = coor_emb.shape[0]
    for j in range(bs):
        _, rvec, T, inliners = cv2.solvePnPRansac(
            objectPoints=coor_emb[j], imagePoints=uv_emb[j],
            cameraMatrix=K_matrix[j],
            distCoeffs=None, flags=cv2.SOLVEPNP_EPNP, confidence=0.9999, reprojectionError=1
        )
        if inliners.shape[0] > n:
            c_mask = np.zeros(inliners.shape[0], dtype=int)
            c_mask[:n] = 1
            np.random.shuffle(c_mask)
            inliners = inliners[c_mask.nonzero()]
        else:
            inliners = np.pad(inliners, (0, n - inliners.shape[0]), 'wrap')
        ncs.append(inliners.T)

    new_choosed = np.stack(ncs, 0)
    return new_choosed



if __name__ == '__main__':
    # from lib.network.KRRN import KRRN
    # estimator = KRRN(9)
    # print(estimator.state_dict().keys())
    a = np.random.randn(125, 1)
    b = np.random.randn(125, 1)
    print(np.stack([a, b], 0).shape)