#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 22:02
# @Author  : yaomy
import torch
import math
import numpy as np
from typing import List

from transforms3d.euler import euler2quat, mat2euler, quat2euler, euler2mat
from transforms3d.quaternions import axangle2quat, mat2quat, qinverse, qmult, quat2mat
from transforms3d.axangles import axangle2mat


def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    """ from gdrnet, differentable
    """
    # translation: Nx3
    # rot_allo: Nx3x3
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )
    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)
    # Apply quaternion for transformation from allocentric to egocentric.
    rot_ego = torch.matmul(rot_allo_to_ego, rot_allo)
    return rot_ego


def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    # print('quat', quat) # Bx4
    # print('norm_quat: ', norm_quat)  # Bx1
    norm_quat = quat / (norm_quat + eps)
    # print('normed quat: ', norm_quat)
    qw, qx, qy, qz = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [1.0 - (yY + zZ), xY - wZ, xZ + wY, xY + wZ, 1.0 - (xX + zZ), yZ - wX, xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=1
    ).reshape(B, 3, 3)

    # rotMat = torch.stack([
    #     qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    #     2 * (qx * qy + qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2 * (qy * qz - qw * qx),
    #     2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz],
    #     dim=1).reshape(B, 3, 3)

    # w2, x2, y2, z2 = qw*qw, qx*qx, qy*qy, qz*qz
    # wx, wy, wz = qw*qx, qw*qy, qw*qz
    # xy, xz, yz = qx*qy, qx*qz, qy*qz

    # rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
    #                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
    #                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def allocentric_to_egocentric(allo_pose, src_type="mat", dst_type="mat"):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing,
    we try to visually correct by rotating back the amount between
    optical center ray and object centroid ray. Another way to solve
    that might be translational variance
    (https://arxiv.org/abs/1807.03247)
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray([0, 0, 1.0])
    if src_type == "mat":
        trans = allo_pose[:3, 3]
    elif src_type == "quat":
        trans = allo_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
            if src_type == "mat":
                ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
            elif src_type == "quat":
                ego_pose[:3, :3] = np.dot(rot_mat, quat2mat(allo_pose[:4]))
        elif dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), angle)
            if src_type == "quat":
                ego_pose[:4] = qmult(rot_q, allo_pose[:4])
            elif src_type == "mat":
                ego_pose[:4] = qmult(rot_q, mat2quat(allo_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:  # allo to ego
        if src_type == "mat" and dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[:4] = mat2quat(allo_pose[:3, :3])
            ego_pose[4:7] = allo_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, :3] = quat2mat(allo_pose[:4])
            ego_pose[:3, 3] = allo_pose[4:7]
        else:
            ego_pose = allo_pose.copy()
    return ego_pose


def egocentric_to_allocentric(ego_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    """ undifferentable
    """
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = ego_pose[:3, 3]
    elif src_type == "quat":
        trans = ego_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    # 两个向量之间的夹角 theta=arccos(a*b/(|a|*|b|))  归一化之后|a|*|b|=1
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount
    if angle > 0:
        if dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, 3] = trans
            # 计算相机主轴和穿过物体中心点射线之间的旋转矩阵
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
            if src_type == "mat":
                allo_pose[:3, :3] = np.dot(rot_mat, ego_pose[:3, :3])
            elif src_type == "quat":
                allo_pose[:3, :3] = np.dot(rot_mat, quat2mat(ego_pose[:4]))
        elif dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), -angle)
            if src_type == "quat":
                allo_pose[:4] = qmult(rot_q, ego_pose[:4])
            elif src_type == "mat":
                allo_pose[:4] = qmult(rot_q, mat2quat(ego_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:
        if src_type == "mat" and dst_type == "quat":
            allo_pose = np.zeros((7,), dtype=ego_pose.dtype)
            allo_pose[:4] = mat2quat(ego_pose[:3, :3])
            allo_pose[4:7] = ego_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
            allo_pose[:3, :3] = quat2mat(ego_pose[:4])
            allo_pose[:3, 3] = ego_pose[4:7]
        else:
            allo_pose = ego_pose.copy()
    return allo_pose
