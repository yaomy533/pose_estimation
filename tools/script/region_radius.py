#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 11:11
# @Author  : yaomy

# --------------------------------------------------------
# 求region每一小片的size
# --------------------------------------------------------


import os
import numpy as np


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

obj_dict = {
            'ape': 1, 'benchvise': 2, 'cam': 4, 'can': 5, 'cat': 6, 'driller': 8,
            'duck': 9, 'eggbox': 10, 'glue': 11, 'holepuncher': 12, 'iron': 13, 'lamp': 14, 'phone': 15,
}

def main():
    root = '/root/Source/ymy_dataset/Linemod_preprocessed'
    for obj, obj_id in obj_dict.items():
        mdl = ply_vtx(os.path.join(root, 'models', f'obj_{obj_id:02d}.ply'))
