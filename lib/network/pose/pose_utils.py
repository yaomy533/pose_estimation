#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/14 15:53
# @Author  : yaomy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_vertical_rot_vec(c1, c2, y, z):
    ##  c1, c2 are weights
    ##  y, x are rotation vectors
    y = y.view(-1)
    z = z.view(-1)
    rot_x = torch.cross(y, z)
    rot_x = rot_x / (torch.norm(rot_x) + 1e-8)
    # cal angle between y and z
    y_z_cos = torch.sum(y * z)
    y_z_theta = torch.acos(y_z_cos)
    theta_2 = c1 / (c1 + c2) * (y_z_theta - math.pi / 2)
    theta_1 = c2 / (c1 + c2) * (y_z_theta - math.pi / 2)
    # first rotate y
    c = torch.cos(theta_1)
    s = torch.sin(theta_1)
    rotmat_y = torch.tensor([[rot_x[0]*rot_x[0]*(1-c)+c, rot_x[0]*rot_x[1]*(1-c)-rot_x[2]*s, rot_x[0]*rot_x[2]*(1-c)+rot_x[1]*s],
                             [rot_x[1]*rot_x[0]*(1-c)+rot_x[2]*s, rot_x[1]*rot_x[1]*(1-c)+c, rot_x[1]*rot_x[2]*(1-c)-rot_x[0]*s],
                             [rot_x[0]*rot_x[2]*(1-c)-rot_x[1]*s, rot_x[2]*rot_x[1]*(1-c)+rot_x[0]*s, rot_x[2]*rot_x[2]*(1-c)+c]]).to(y.device)
    new_y = torch.mm(rotmat_y, y.view(-1, 1))
    # then rotate z
    c = torch.cos(-theta_2)
    s = torch.sin(-theta_2)
    rotmat_z = torch.tensor([[rot_x[0] * rot_x[0] * (1 - c) + c, rot_x[0] * rot_x[1] * (1 - c) - rot_x[2] * s,
                              rot_x[0] * rot_x[2] * (1 - c) + rot_x[1] * s],
                             [rot_x[1] * rot_x[0] * (1 - c) + rot_x[2] * s, rot_x[1] * rot_x[1] * (1 - c) + c,
                              rot_x[1] * rot_x[2] * (1 - c) - rot_x[0] * s],
                             [rot_x[0] * rot_x[2] * (1 - c) - rot_x[1] * s,
                              rot_x[2] * rot_x[1] * (1 - c) + rot_x[0] * s, rot_x[2] * rot_x[2] * (1 - c) + c]]).to(
        z.device)

    new_z = torch.mm(rotmat_z, z.view(-1, 1))
    return new_y.view(-1), new_z.view(-1)


def get_rot_mat_y_first(y, x):
    # poses
    y = F.normalize(y, p=2, dim=-1)  # bx3
    z = torch.cross(x, y, dim=-1)  # bx3
    z = F.normalize(z, p=2, dim=-1)  # bx3
    x = torch.cross(y, z, dim=-1)  # bx3

    # (*,3)x3 --> (*,3,3)
    return torch.stack((x, y, z), dim=-1)  # (b,3,3)

def get_rot_vec_vert_batch(c1, c2, y, z):
    bs = c1.shape[0]
    new_y = y
    new_z = z
    for i in range(bs):
        new_y[i, ...], new_z[i, ...] = get_vertical_rot_vec(c1[i, ...], c2[i, ...], y[i, ...], z[i, ...])
    return new_y, new_z
