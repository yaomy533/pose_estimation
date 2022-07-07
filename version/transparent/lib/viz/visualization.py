#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 12:59
# @Author  : yaomy

import torch
import cv2.cv2 as cv2
import numpy as np


class DrawPred:
    def __init__(self, model, dataloader, loss_func=None):
        self.model = model
        self.dataloader = dataloader
        self.loss_func = loss_func
        self.dataiter = self.dataloader.__iter__()

    def get_pred(self, data):
        pass

    def load_data(self):
        return self.dataiter.__next__()

    @staticmethod
    def perspective_transformation(pt, intrinsic):
        """
        点云投影到像素点, 射影变换
        """
        if len(intrinsic) == 3:
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]
        elif len(intrinsic) == 4:
            fx = intrinsic[0]
            fy = intrinsic[1]
            cx = intrinsic[2]
            cy = intrinsic[3]
        else:
            raise ValueError('The format of the  intrinsic is incorrect')
        u = pt[:, 0]*fx/pt[:, 2]+cx
        v = pt[:, 1]*fy/pt[:, 2]+cy
        pixels = np.stack((u, v), axis=1).astype(np.int32)
        return pixels

    @staticmethod
    def rt2extrinsic(r, t):
        extrinsic = torch.eye(4).to(r.device)
        extrinsic[:3, :3] = r
        T = t.unsqueeze(dim=0).reshape(3, 1)
        extrinsic[:3, 3] = T[:, 0]
        return extrinsic

    @staticmethod
    def draw_axis(img, axes):
        """
        :param img: [H, W, C]
        :param axes: [4, 3]
        """
        # draw axes
        img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
        img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
        img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3)
        return img

    @staticmethod
    def draw_bbox(img, imgpts, color):
        imgpts = np.int32(imgpts).reshape(-1, 2)
        # draw ground layer in darker color
        color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
        for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)

        # draw pillars in blue color
        color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

        # finally, draw top layer in color
        for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
            img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)

    @staticmethod
    def draw_circle(img, p2ds, r=1, color=(255, 0, 0)):
        """
        :param color: (int, int, int)
        :param img:[h, w, 3]
        :param p2ds: [k, 2, N] k is number of obj
        :param r: 1
        :return:img [h, w, 3]
        """
        h, w = img.shape[0], img.shape[1]
        for u, v in zip(p2ds[:, 0], p2ds[:, 1]):
            u = np.clip(u, 0, w)
            v = np.clip(v, 0, h)
            img = cv2.circle(img, (u, v), r, color, -1)
        return img
