#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/26 20:11
# @Author  : yaomy
import numpy as np
import math
import torch


def get_rotation(x_, y_, z_):
    # print(math.cos(math.pi/2))
    x = float(x_ / 180) * math.pi
    y = float(y_ / 180) * math.pi
    z = float(z_ / 180) * math.pi
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]])

    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))


def trans_3d(pc, Rt, Tt):
    Tt = np.reshape(Tt, (3, 1))
    pcc = np.zeros((4, pc.shape[0]), dtype=np.float32)
    pcc[0:3, :] = pc.T
    pcc[3, :] = 1

    TT = np.zeros((3, 4), dtype=np.float32)
    TT[:, 0:3] = Rt
    TT[:, 3] = Tt[:, 0]
    trans = np.dot(TT, pcc)
    return trans


def rotation_axis(R, a=15):
    r""" from FSNet 感觉没啥用
    """
    Rt = R.reshape(3, 3)
    Tt_c = np.array([0, 0, 0]).T
    corners_ = np.array([[0, 0, 0], [0, 200, 0], [200, 0, 0]])
    rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
    corners = trans_3d(corners_, np.dot(rm, Rt), Tt_c).T.flatten()
    return corners


def quaternion_matrix_torch(quaternion):
    """
    Return homogeneous rotation matrix from quaternion.
    undifferentiable
    """
    _TEPS = torch.finfo(torch.float).eps * 4.0
    q = quaternion.clone().detach().type(torch.float64)
    device = q.device
    n = torch.dot(q, q)
    if n < _TEPS:
        return torch.eye(4)
    q *= torch.sqrt(2.0 / n)
    q = torch.outer(q, q)
    return torch.tensor([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]], device=device)


if __name__ == '__main__':
    import cv2.cv2 as cv2
    r0 = np.random.randn(3)
    R0, _ = cv2.Rodrigues(r0)
    print(rotation_axis(R0).shape)
