#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 14:03
# @Author  : yaomy
import torch
import numpy as np
from typing import List
from kornia.geometry.conversions import rotation_matrix_to_quaternion
import kornia as kn
import torch.nn.functional as F


class Metric:
    def __init__(self, sys: List):
        self.sys = sys

    def cal_adds_cuda(
        self, pred, target, idx
    ):
        """follow pvn3d https://github.com/ethnhe/PVN3D/tree/pytorch-1.5
        """
        assert pred.dim() == target.dim() == 2
        add_dis = torch.mean(torch.linalg.norm(pred - target, dim=1))
        if idx in self.sys:
            N, _ = pred.size()
            # cuda 加速计算
            pd = pred.cuda().view(1, N, 3).repeat(N, 1, 1)
            gt = target.cuda().view(N, 1, 3).repeat(1, N, 1)
            dis = torch.norm(pd - gt, dim=2)
            mdis = torch.min(dis, dim=1)[0]
            adds_dis = torch.mean(mdis)
            add_dis = adds_dis
        else:
            adds_dis = add_dis

        return add_dis.item(), adds_dis.item()

    def cal_auc(self, add_dis, max_dis=0.1):
        """
        :param add_dis: ADD-S距离
        :param max_dis:
        :return: AUC
        """
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1, n)), dtype=np.float32) / n
        aps = self.voc_ap(D, acc)
        return aps * 100.0

    @staticmethod
    def voc_ap(rec, prec):
        idx = np.where(rec != np.inf)
        if len(idx[0]) == 0:
            return 0
        rec = rec[idx]
        prec = prec[idx]
        mrec = np.array([0.0] + list(rec) + [0.1])
        mpre = np.array([0.0] + list(prec) + [prec[-1]])
        for i in range(1, prec.shape[0]):
            mpre[i] = max(mpre[i], mpre[i - 1])
        i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
        ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) * 10
        return ap

    @staticmethod
    def angular_distance(q1, q2, eps: float = 1e-7):
        if q1.dim == 3 or q1.size(-1) == 3:
            q1 = rotation_matrix_to_quaternion(q1.contiguous(), order=kn.geometry.conversions.QuaternionCoeffOrder.WXYZ)
        if q2.dim == 3 or q2.size(-1) == 3:
            q2 = rotation_matrix_to_quaternion(q2.contiguous(), order=kn.geometry.conversions.QuaternionCoeffOrder.WXYZ)

        q1 = normalize(q1)
        q2 = normalize(q2)

        dot = q1 @ q2.t()
        dist = 2 * acos_safe(dot.abs(), eps=eps)
        rotation_dis = dist / torch.pi * 180.
        return rotation_dis

    @staticmethod
    def translation_distance(t1, t2):
        return torch.norm(t1 - t2, dim=-1)


def normalize(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (x, y, z, w) format.

    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.

    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.

    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape (*, 4). Got {}".format(
                quaternion.shape))
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


def acos_safe(t, eps: float = 1e-7):
    return torch.acos(torch.clamp(t, min=-1.0 + eps, max=1.0 - eps))



if __name__ == '__main__':
    a = torch.tensor([1.0])
    print(type(a).__name__)