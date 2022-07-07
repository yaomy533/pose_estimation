#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/7 19:20
# @Author  : yaomy

import torch
import torch.nn as nn


def best_batch_fit_transform(A, B):
    """
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: cxNxm torch tensor of corresponding points, usually points on mdl
        B: cxNxm torch tensor of corresponding points, usually points on camera axis
    Returns:
    T: cx(m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: cxmxm rotation matrix
    t: cxmx1 translation vector
    """

    A = A.float()
    B = B.float()
    assert A.shape == B.shape
    # get number of dimensions
    c, n, m = A.shape
    # translate points to their centroids
    centroid_A = torch.mean(A, dim=1)
    centroid_B = torch.mean(B, dim=1)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matirx
    H = AA.permute(0, 2, 1) @ BB
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.permute(0, 2, 1) @ U.permute(0, 2, 1)

    # special reflection case
    det = torch.linalg.det(R)
    idxs = torch.where(det < 0)[0]
    if idxs.size(0) != 0:
        Vt[idxs, m-1, :] *= -1
        R = Vt.permute(0, 2, 1) @ U.permute(0, 2, 1)

    t = centroid_B - (R @ centroid_A.view(c, m, 1)).squeeze()

    return R, t, centroid_A, centroid_B


class LeastSquaresFcn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, target):
        bs, num, _ = inputs.shape
        R, t, centroid_A, centroid_B = best_batch_fit_transform(inputs, target)
        ctx.save_for_backward(inputs, target, R, t)
        return R, t

    @staticmethod
    def backward(ctx, grad_r, grad_t):
        # q q_ [c, m, 3]
        # R [c, 3, 3]
        # R->input
        torch.set_grad_enabled(True)
        inputs, target, R, t = ctx.saved_tensors

        inputs, target, R, t = \
            inputs.detach().clone().detach().contiguous().requires_grad_(), \
            target.detach().clone().detach().contiguous().requires_grad_(),\
            R.detach().clone().detach().contiguous().requires_grad_(),\
            t.detach().clone().detach().contiguous().requires_grad_(),
        num_c, num_p, _ = inputs.shape
        centroid_A = torch.mean(inputs, dim=1)
        centroid_B = torch.mean(target, dim=1)

        # q q_ [c, m, 3, 1]
        q = (inputs - centroid_A).view(num_c, num_p, 3)
        q_ = (target - centroid_B).view(num_c, num_p, 3)

        # q = (inputs - centroid_A).view(num_c, num_p, 3, 1)
        # q_ = (target - centroid_B).view(num_c, num_p, 3, 1)
        # # [1000, 1000, 3, 3]
        # F = q @ q.permute(0, 1, 3, 2) @ R + R.permute(0, 2, 1) @ q @ q.permute(0, 1, 3, 2) - 2*q@q_.permute(0, 1, 3, 2)
        # # [1000, 3, 3]
        # F_r = torch.autograd.grad(F, R, torch.ones_like(F), retain_graph=True)[0].clone()
        # inv_F_r = torch.linalg.inv(F_r[0])
        # # [1000, 1000, 3]
        # F_q = torch.autograd.grad(F, q, torch.ones_like(F))[0].clone().squeeze()
        #
        # q = q.squeeze()
        # # [1000, 3, 3]
        #

        # 直接求二阶偏导数
        f = (q_-q@R.permute(0, 2, 1)).permute(0, 2, 1) @ (q_-q@R.permute(0, 2, 1))
        F = torch.autograd.grad(f, R, torch.ones_like(f), retain_graph=True, create_graph=True)[0]
        F_r = torch.autograd.grad(F, R, torch.ones_like(F), retain_graph=True, create_graph=True)[0]
        F_q = torch.autograd.grad(F, q, torch.ones_like(F), retain_graph=True, create_graph=True)[0]
        # Frs = torch.tensor([]).cuda()
        # qs = torch.tensor([]).cuda()
        # for anygrad in F.contiguous().view(num_c, -1).T:  # torch.autograd.grad返回的是元组
        #     Frs = torch.cat((Frs, torch.autograd.grad(anygrad.T.sum(), R, retain_graph=True, create_graph=True)[0]))
        #     qs = torch.cat((qs, torch.autograd.grad(anygrad.T.sum(), q, retain_graph=True, create_graph=True)[0]))
        #
        # F_r = Frs.view(num_c, 3, 3)
        # F_q = Frs.view(num_c, num_p, 3)

        inv_F_r = torch.linalg.pinv(F_r)
        derivative = -1 * (F_q @ inv_F_r.permute(0, 2, 1))
        grad_r2inputes = torch.autograd.grad(q, inputs, derivative)[0]
        # T -> R, input
        t = centroid_B - (R @ centroid_A.view(num_c, 3, 1)).squeeze()

        grad_t2r = torch.autograd.grad(t, R, grad_t, retain_graph=True)[0].clone()
        grad_t2input = torch.autograd.grad(t, inputs, grad_t)[0].clone()

        input_grad = grad_r2inputes @ (grad_r+grad_t2r).permute(0, 2, 1) + grad_t2input

        return input_grad, None


class LeastSquaresLayer(nn.Module):
    def __init__(self):
        super(LeastSquaresLayer, self).__init__()

    def forward(self, inputs, target):
        return LeastSquaresFcn.apply(inputs, target)


if __name__ == '__main__':
    a = torch.tensor([[4.0698, 4.0698, 4.0698],
        [7.0041, 7.0041, 7.0041],
        [0.5831, 0.5831, 0.5831]], device='cuda:0')

    print(torch.linalg.det(a))
