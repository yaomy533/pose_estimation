import torch
import math
import random
import numpy as np
import cv2.cv2 as cv2
import torch.nn.functional as F

from typing import Union, Tuple, Dict, List
from pykeops.torch import generic_argkmin


class Metric:
    def __init__(self, sys: List):
        self.sys = sys
        self.knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')

    def cal_dis(self, pred, target, idx):
        """
        :param pred: [n, 3]
        :param target: [n, 3]
        :param idx: cls of obj
        :return: add_dis, adds_dis
        """
        inds = self.knn(pred.float(), target.float())
        target_kp = torch.index_select(target, 0, inds.view(-1))
        adds_dis = torch.mean(torch.linalg.norm((pred - target_kp), dim=1))
        add_dis = torch.mean(torch.linalg.norm(pred - target, dim=1))
        if idx in self.sys:
            add_dis = adds_dis
        return add_dis.item(), adds_dis.item()

    def cal_adds_cuda(
            self, pred, target, idx
    ):
        """follow pvn3d https://github.com/ethnhe/PVN3D/tree/pytorch-1.5
        """
        add_dis = torch.mean(torch.linalg.norm(pred - target, dim=1))
        N, _ = pred.size()
        pd = pred.view(1, N, 3).repeat(N, 1, 1)
        gt = target.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        adds_dis = torch.mean(mdis)
        if idx in self.sys:
            add_dis = adds_dis
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


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def load_obj(filename_obj, normalization=False, texture_size=4, load_texture=False,
             texture_wrapping='REPEAT', use_bilinear=True):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[0][None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[0][None, :] / 2

    return vertices, faces


def sample_points_from_mesh(path, n_pts, with_normal=False, fps=False, ratio=2):
    """ Uniformly sampling points from mesh model.

    Args:
        path: path to OBJ file.
        n_pts: int, number of points being sampled.
        with_normal: return points with normal, approximated by mesh triangle normal
        fps: whether to use fps for post-processing, default False.
        ratio: int, if use fps, sample ratio*n_pts first, then use fps to sample final output.

    Returns:
        points: n_pts x 3, n_pts x 6 if with_normal = True

    """
    vertices, faces = load_obj(path)
    if fps:
        points = uniform_sample(vertices, faces, ratio * n_pts, with_normal)
        pts_idx = farthest_point_sampling(points[:, :3], n_pts)
        points = points[pts_idx]
    else:
        points = uniform_sample(vertices, faces, n_pts, with_normal)
    return points


def uniform_sample(vertices, faces, n_samples, with_normal=False):
    """ Sampling points according to the area of mesh surface.

    """
    sampled_points = np.zeros((n_samples, 3), dtype=float)
    normals = np.zeros((n_samples, 3), dtype=float)
    faces = vertices[faces]
    vec_cross = np.cross(faces[:, 1, :] - faces[:, 0, :],
                         faces[:, 2, :] - faces[:, 0, :])
    face_area = 0.5 * np.linalg.norm(vec_cross, axis=1)
    cum_area = np.cumsum(face_area)
    for i in range(n_samples):
        face_id = np.searchsorted(cum_area, np.random.random() * cum_area[-1])
        sampled_points[i] = random_point(faces[face_id, :, :])
        normals[i] = vec_cross[face_id]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if with_normal:
        sampled_points = np.concatenate((sampled_points, normals), axis=1)
    return sampled_points


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


# def randomly_selected_mask(mask: torch.Tensor, num):
#     """错误的代码，没有实现按照顺序选取
#     """
#     device = mask.device
#     _, indices_all = torch.sort(mask, dim=-1, descending=True)
#     if indices_all.size(-1) < num:
#         indices_all = F.pad(indices_all, (num-indices_all.size(-1), num-indices_all.size(-1)))
#     index = torch.LongTensor(random.sample(range(indices_all.size(-1)), num)).to(device)
#     indices = torch.index_select(torch.sort(indices_all, dim=-1, descending=True)[0], -1, index)
#     return indices.unsqueeze(dim=1)

def randomly_selected_mask(mask: torch.Tensor, num, negtive_falg=False):
    """
    """
    bs = mask.size(0)
    if negtive_falg:
        num_pos = int(num/3)
    else:
        num_pos = num
    num_neg = int(num/2)
    positive = torch.where(mask.flatten() >= 0.7)[0]
    negative = torch.where(mask.flatten() < 0.3)[0]

    if positive.size(-1) < num_pos:
        positive = torch.where(mask.flatten() >= 0.5)[0]
        if positive.size(-1) < num_pos:
            positive = F.pad(positive, (num_pos - positive.size(-1), num_pos - positive.size(-1)))

    if negative.size(-1) < num_neg:
        negative = torch.where(mask.flatten() < 0.5)[0]
        if negative.size(-1) < num_neg:
            negative = F.pad(negative, (num_neg - negative.size(-1), num_neg - negative.size(-1)))

    perm1 = torch.randperm(positive.size(-1), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.size(-1), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]

    if negtive_falg:
        return torch.cat([pos_idx, neg_idx]).view(bs, 1, -1)
    else:
        return pos_idx.view(bs, 1, -1)


def batch_index_select(x, dim, idx):
    return torch.cat([torch.index_select(a, dim, i).unsqueeze(0) for a, i in zip(x, idx)])


if __name__ == '__main__':
    import torchvision
    print(torchvision.models.detection.__file__)
