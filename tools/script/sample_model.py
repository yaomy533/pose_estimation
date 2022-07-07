#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 13:43
# @Author  : yaomy
import torch
import os
from pathlib import Path
import random
import numpy as np
import pickle
root = '/root/Source/ymy_dataset/Linemod_preprocessed'

obj_dict = {
            'ape': 1, 'benchvise': 2, 'cam': 4, 'can': 5, 'cat': 6, 'driller': 8,
            'duck': 9, 'eggbox': 10, 'glue': 11, 'holepuncher': 12, 'iron': 13, 'lamp': 14, 'phone': 15,
}


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


def farthest_point_sampling(points, n_samples):
    """ Farthest point sampling, the first point is fixed at the 0th index.

    """
    selected_pts = np.zeros((n_samples,), dtype=int)
    dist_mat = pairwise_distance(points, points)
    # start from first point
    pt_idx = 0
    dist_to_set = dist_mat[:, pt_idx]
    for i in range(n_samples):
        selected_pts[i] = pt_idx
        dist_to_set = np.minimum(dist_to_set, dist_mat[:, pt_idx])
        pt_idx = np.argmax(dist_to_set)
    return selected_pts


def pairwise_distance(A, B):
    """ Compute pairwise distance of two point clouds.point

    Args:
        A: n x 3 numpy array
        B: m x 3 numpy array

    Return:
        C: n x m numpy array

    """
    diff = A[:, :, None] - B[:, :, None].T
    C = np.sqrt(np.sum(diff ** 2, axis=1))

    return C


def random_point(face_vertices):
    """ Sampling point using Barycentric coordiante.

    """
    r1, r2 = np.random.random(2)
    sqrt_r1 = np.sqrt(r1)
    point = (1 - sqrt_r1) * face_vertices[0, :] + \
            sqrt_r1 * (1 - r2) * face_vertices[1, :] + \
            sqrt_r1 * r2 * face_vertices[2, :]

    return point

def main():
    for obj, obj_id in obj_dict.items():
        print('load model: ', obj)
        mdl = ply_vtx(os.path.join(root, 'models', f'obj_{obj_id:02d}.ply'))
        sm_mdl_idx = farthest_point_sampling(mdl, 5000)
        sm_mdl = mdl[sm_mdl_idx]
        with open(os.path.join(root, 'models', f'obj_{obj_id:02d}.pkl'), 'wb') as fp:
            pickle.dump(sm_mdl, fp)

if __name__ == '__main__':
    main()
