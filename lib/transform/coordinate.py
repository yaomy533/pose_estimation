#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 18:49
# @Author  : yaomy
import cv2
import torch
import torch.nn.functional as F
import numpy as np


def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)
    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)
    return dst_img


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)



def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def uvd2cloud(uvd, K):
    """ Differentiable bath back projection
    :param uvd: [bs, n, 3]
    :param K: [3, 3]
    :return: xyz [bs, n, 3]
    """
    cam_fx, cam_fy, cam_cx, cam_cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uvd[..., 0] - cam_cx) * uvd[..., 2] / cam_fx
    y = (uvd[..., 1] - cam_cy) * uvd[..., 2] / cam_fy
    z = uvd[..., 2]
    return torch.cat([x, y, z], dim=-1)


def normalize_vector(v):
    # bxn
    # batch = v.shape[0]
    # v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    # v_mag = torch.max(v_mag, torch.FloatTensor([1e-8]).to(v))
    # v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    # v = v / v_mag
    v = F.normalize(v, p=2, dim=1)
    return v


def cross_product(u, v):
    # u, v bxn

    batch = u.size(0)
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # bx3

    return out


def ortho6d_to_mat_batch(poses):
    # poses bx6
    # poses
    x_raw = poses[:, 0:3]  # bx3
    y_raw = poses[:, 3:6]  # bx3

    x = normalize_vector(x_raw)  # bx3
    z = cross_product(x, y_raw)  # bx3
    z = normalize_vector(z)  # bx3
    y = cross_product(z, x)  # bx3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # bx3x3
    return matrix

