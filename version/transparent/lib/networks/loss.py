#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 21:43
# @Author  : yaomy
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch.nn.functional as F

from lib.transform.allocentric import allo_to_ego_mat_torch
from pykeops.torch import generic_argkmin
# import pykeops
# pykeops.clean_pykeops()
# knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')


def l2_loss(pred, target, reduction="mean"):
    assert pred.size() == target.size() and target.numel() > 0
    assert pred.size()[0] == target.size()[0]
    batch_size = pred.size()[0]
    loss = torch.norm((pred - target).view(batch_size, -1), p=2, dim=1, keepdim=True)
    # loss = torch.sqrt(torch.sum(((pred - target)** 2).view(batch_size, -1), 1))
    # print(loss.shape)
    """
    _mse_loss = nn.MSELoss(reduction='none')
    loss_mse = _mse_loss(pred, target)
    print('l2 from mse loss: {}'.format(
        torch.sqrt(
            torch.sum(
                loss_mse.view(batch_size, -1),
                1
            )
        ).mean()))
    """
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class L2Loss(_Loss):
    """ l2 loss from GDRNet
    """
    def __init__(self, reduction="mean", loss_weight=1.0):
        super(L2Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = self.loss_weight * l2_loss(pred, target, reduction=self.reduction)
        return loss


class FocalLoss(_Loss):
    """ mask loss from PVN3D
    """
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        if target.dim() > 2:
            target = target.view(target.size(0), target.size(1), -1)  # N,C,H,W => N,C,H*W
            target = target.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target = target.contiguous().view(-1, target.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


# def loss_calculation(
#         pred_r, pred_t, pred_c,
#         target, model_points, idx,
#         points, w, refine, num_point_mesh, sym_list,
#         axis, target_r
#
# ):
#
#     """ ADD(S)-based loss from Densefusion, 但是这里位移是绝对位移不再是偏移量， 去掉了refine
#     加入了对旋转轴的约束
#     """
#     knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')
#     bs, num_p, _ = pred_c.size()
#     pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
#     base = torch.cat((
#         (1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),
#         (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1),
#         (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1),
#         (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1),
#         (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),
#         (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1),
#         (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1),
#         (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1),
#         (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)
#     ), dim=2).contiguous().view(bs * num_p, 3, 3)
#
#     model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
#     target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, num_point_mesh, 3)
#
#     # all->eg [N, 3, 3]
#     base = allo_to_ego_mat_torch(pred_t.contiguous().view(bs * num_p, 3), base)
#     axis = axis.repeat(bs * num_p, 1)
#     target_r = target_r.repeat(bs * num_p, 1, 1)
#     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#     loss_axis = \
#         axis[:, 0] * cos(base.transpose(1, 2).contiguous()[:, 0], target_r.transpose(1, 2)[:, 0]) \
#         + axis[:, 1] * cos(base.transpose(1, 2).contiguous()[:, 1], target_r.transpose(1, 2)[:, 1]) \
#         + axis[:, 2] * cos(base.transpose(1, 2).contiguous()[:, 2], target_r.transpose(1, 2)[:, 2])
#
#     pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
#     pred_c = pred_c.contiguous().view(bs * num_p)
#
#     pred = torch.add(torch.bmm(model_points, base.transpose(1, 2).contiguous()), pred_t)
#
#     if not refine:
#         if idx in sym_list:
#             target = target[0].transpose(1, 0).contiguous().view(3, -1)
#             pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
#             # pred: torch.Size([3, 250000]) target: torch.Size([3, 500])
#
#             inds = knn(pred.T.contiguous(), target.T.contiguous())
#             target = torch.index_select(target, 1, inds.view(-1))
#             target = target.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
#             pred = pred.view(3, bs * num_p, num_point_mesh).permute(1, 2, 0).contiguous()
#
#     dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)
#     loss = torch.mean((dis * pred_c - w * torch.log(pred_c)), dim=0)
#
#     pred_c = pred_c.view(bs, num_p)
#     how_max, which_max = torch.max(pred_c, 1)
#     dis = dis.view(bs, num_p)
#
#     return loss, dis[0][which_max[0]]


def loss_fn_cosine(input_vec, target_vec, reduction='sum'):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_cos = 1.0 - cos(input_vec, target_vec)

    mask_invalid_pixels = torch.all(target_vec == 0., dim=1)
    loss_cos[mask_invalid_pixels] = 0.0
    loss_cos_sum = loss_cos.sum()
    total_valid_pixels = (~mask_invalid_pixels).sum().double()
    error_output = loss_cos_sum / total_valid_pixels

    if reduction == 'elementwise_mean':
        loss_cos = error_output
    elif reduction == 'sum':
        loss_cos = loss_cos_sum
    elif reduction == 'none':
        loss_cos = loss_cos
    else:
        raise Exception(
            'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())
    return loss_cos


class NormalLoss(_Loss):
    """ from cleargrasp
    """
    def __init__(self, reduction='elementwise_mean'):
        super(NormalLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_n, gt_n):
        nl = loss_fn_cosine
        loss_normal = nl(pred_n, gt_n.double(), reduction=self.reduction)
        return loss_normal


class PossLoss(_Loss):
    def __init__(self, num_point_mesh, sym_list, knn):
        super(PossLoss, self).__init__()
        self.num_point_mesh = num_point_mesh
        self.sym_list = sym_list
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')
        self.knn = knn
        self._EPS = torch.finfo(torch.float).eps * 4.0

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, refine, w, axis, target_r):
        bs, num_p, _ = pred_c.size()
        base = self.predr2rotation(pred_r, bs, num_p)

        # all->eg [N, 3, 3]
        base = allo_to_ego_mat_torch(pred_t.contiguous().view(bs * num_p, 3), base)

        model_points = model_points.view(bs, 1, self.num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, self.num_point_mesh, 3)
        target = target.view(bs, 1, self.num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs * num_p, self.num_point_mesh, 3)

        target_r = target_r.repeat(bs * num_p, 1, 1)
        pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)
        pred_c = pred_c.contiguous().view(bs * num_p)
        axis = axis.repeat(bs * num_p, 1)

        pred = torch.add(torch.bmm(model_points, base.transpose(1, 2).contiguous()), pred_t)

        if not refine:
            if idx in self.sym_list:
                target = target[0].transpose(1, 0).contiguous().view(3, -1)
                pred = pred.permute(2, 0, 1).contiguous().view(3, -1)
                # pred: torch.Size([3, 250000]) target: torch.Size([3, 500])

                inds = self.knn(pred.T.contiguous(), target.T.contiguous())
                target = torch.index_select(target, 1, inds.view(-1))
                target = target.view(3, bs * num_p, self.num_point_mesh).permute(1, 2, 0).contiguous()
                pred = pred.view(3, bs * num_p, self.num_point_mesh).permute(1, 2, 0).contiguous()

        dis = torch.mean(torch.norm((pred - target), dim=2), dim=1)

        loss = torch.mean((dis * pred_c - w * torch.log(pred_c+1e-8)), dim=0)

        loss_axis = \
            axis[:, 0] * (1. - self.cos(base.transpose(1, 2).contiguous()[:, 0], target_r.transpose(1, 2)[:, 0])) \
            + axis[:, 1] * (1. - self.cos(base.transpose(1, 2).contiguous()[:, 1], target_r.transpose(1, 2)[:, 1])) \
            + axis[:, 2] * (1. - self.cos(base.transpose(1, 2).contiguous()[:, 2], target_r.transpose(1, 2)[:, 2]))

        # loss_rotatoion = torch.mean((pred_c*loss_axis - w * torch.log(pred_c.clamp_(min=1e-7))), dim=0)
        loss_rotatoion = torch.mean((pred_c*loss_axis - w * torch.log(pred_c+1e-8)), dim=0)

        pred_c = pred_c.view(bs, num_p)
        how_max, which_max = torch.max(pred_c, 1)
        dis = dis.view(bs, num_p)

        return loss, dis[0][which_max[0]], loss_rotatoion

    @staticmethod
    def predr2rotation(pred_r, bs, num_p):
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))
        base = torch.cat((
            (1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1),
            (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs, num_p, 1),
            (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),
            (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0]).view(bs, num_p, 1),
            (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1),
            (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),
            (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs, num_p, 1),
            (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs, num_p, 1),
            (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)
        ), dim=2).contiguous().view(bs * num_p, 3, 3)
        return base


class Loss(_Loss):
    def __init__(self, num_point_mesh, sym_list, **kwargs):
        super(Loss, self).__init__()
        self.num_point_mesh = num_point_mesh
        self.sym_list = sym_list
        # default: loss_wieght = {'distance': 1.0, 'normal': 1.0, 'depth': 1.0, 'mask': 1.0, 'rotation': 1.0}
        self.loss_weight = kwargs['loss_weight']
        if 'knn' in kwargs.keys():
            self.knn = kwargs['knn']
        else:
            self.knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')
        # loss init
        self.pose_loss_function = PossLoss(num_point_mesh, sym_list, self.knn)
        self.mask_loss_function = nn.L1Loss(reduction="mean")
        self.boundary_loss_function = nn.L1Loss(reduction="mean")
        # self.depth_loss_function = L2Loss()
        # self.depth_loss_function = nn.L1Loss(reduction="mean")
        self.depth_loss_function = nn.SmoothL1Loss(reduction="mean")
        self.normal_loss_function = NormalLoss('elementwise_mean')

    def forward(
            self,
            pred_r, pred_t, pred_c, pred_n, pred_d, pred_m, pred_b, choose,
            target, model_points, idx, w,
            gt_n, gt_d, gt_m, axis, gt_r, gt_b,
            points=None, refine=None
    ):
        # loss_add, distance = loss_calculation(
        #     pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine,
        #     self.num_point_mesh, self.sym_list,
        #     axis=axis,
        #     target_r=gt_r
        # )
        bs = pred_d.size(0)
        loss_add, distance, loss_r = self.pose_loss_function(
            pred_r, pred_t, pred_c, target, model_points, idx, refine, w, axis, gt_r
        )

        loss_n = self.normal_loss_function(pred_n, gt_n)
        loss_m = self.mask_loss_function(pred_m, gt_m)

        loss_b = self.boundary_loss_function(pred_b, gt_b)

        # pred_d = torch.gather(pred_d.contiguous().view(bs, 1, -1), -1, choose)
        # gt_d = torch.gather(gt_d.contiguous().view(bs, 1, -1), -1, choose)
        loss_d = self.depth_loss_function(pred_d, gt_d)

        loss = self.loss_weight['distance']*loss_add\
            + self.loss_weight['normal']*loss_n \
            + self.loss_weight['depth']*loss_d \
            + self.loss_weight['mask']*loss_m \
            + self.loss_weight['rotation']*loss_r \
            + self.loss_weight['boundary']*loss_b

        loss_dict = {
            'all_loss': loss,
            'distance': distance,
            'loss_add': loss_add,
            'loss_r': loss_r,
            'loss_n': loss_n,
            'loss_m': loss_m,
            'loss_d': loss_d,
            'loss_b': loss_b
        }

        return loss, loss_dict


class MultiLoss(_Loss):
    def __init__(self, num_point_mesh, sym_list, **kwargs):
        super(MultiLoss, self).__init__()
        self.num_point_mesh = num_point_mesh
        self.sym_list = sym_list
        # default: loss_wieght = {'distance': 1.0, 'normal': 1.0, 'depth': 1.0, 'mask': 1.0, 'rotation': 1.0}
        self.loss_weight = kwargs['loss_weight']
        # loss init
        if 'knn' in kwargs.keys():
            self.knn = kwargs['knn']
        else:
            self.knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')
        self.pose_loss_function = PossLoss(num_point_mesh, sym_list, self.knn)
        self.mask_loss_function = nn.L1Loss(reduction="mean")
        self.depth_loss_function = nn.SmoothL1Loss(reduction="mean")
        self.normal_loss_function = NormalLoss('elementwise_mean')

    def forward(
            self,
            pred_rs, pred_ts, pred_cs, pred_ns, pred_ds, pred_ms,
            targets, model_points, idxs, w,
            gt_ns, gt_ds, gt_ms, axises, gt_rs
    ):
        # loss_add_list = list()
        # distance_list = list()
        # pred_c_list = list()
        # for pred_r, pred_t, pred_c, target, model_points, idx, axis, gt_r in zip(
        #         pred_rs, pred_ts, pred_cs, targets, model_points, idxs, axises, gt_rs
        # ):
        #     loss_add, distance, loss_r = self.pose_loss_function(
        #         pred_r, pred_t, pred_c, target, model_points, idx, False, w, axis, gt_r
        #     )
        #     loss_add_list.append(loss_add)
        #     distance_list.append(distance_list)
        #     pred_c_list.append(pred_c_list)
        loss_add, distance, loss_r = self.batch_pose_loss(
            pred_rs, pred_ts, pred_cs, targets, model_points, idxs, False, w, axises, gt_rs
        )
        loss_n = self.normal_loss_function(pred_ns, gt_ns)
        loss_m = self.mask_loss_function(pred_ms, gt_ms)

        # pred_d = torch.gather(pred_d.contiguous().view(bs, 1, -1), -1, choose)
        # gt_d = torch.gather(gt_d.contiguous().view(bs, 1, -1), -1, choose)
        loss_d = self.depth_loss_function(pred_ds, gt_ds)

        loss = self.loss_weight['distance']*loss_add\
            + self.loss_weight['normal']*loss_n \
            + self.loss_weight['depth']*loss_d \
            + self.loss_weight['mask']*loss_m \
            + self.loss_weight['rotation']*loss_r

        loss_dict = {
            'all_loss': loss,
            'distance': distance,
            'loss_add': loss_add,
            'loss_r': loss_r,
            'loss_n': loss_n,
            'loss_m': loss_m,
            'loss_d': loss_d,
        }

        return loss, loss_dict

    @property
    def loss_dict(self):
        cc = torch.tensor([0.])
        return {
            'all_loss': cc,
            'distance': cc,
            'loss_add': cc,
            'loss_r': cc,
            'loss_n': cc,
            'loss_m': cc,
            'loss_d': cc,
        }

    def batch_pose_loss(self, pred_rs, pred_ts, pred_cs, targets, model_points, idxs, refine, w, axises, gt_rs):
        """batch add-s loss
        """
        loss_add_list = list()
        distance_list = list()
        loss_r_list = list()

        for pred_r, pred_t, pred_c, target, model_points, idx, axis, gt_r in zip(
                pred_rs, pred_ts, pred_cs, targets, model_points, idxs, axises, gt_rs
        ):
            loss_add, distance, loss_r = self.pose_loss_function(
                pred_r.unsqueeze(dim=0), pred_t.unsqueeze(dim=0), pred_c.unsqueeze(dim=0), target.unsqueeze(dim=0),
                model_points.unsqueeze(dim=0), idx.unsqueeze(dim=0), refine, w, axis.unsqueeze(dim=0), gt_r.unsqueeze(dim=0)
            )
            loss_add_list.append(loss_add.view(1))
            distance_list.append(distance.view(1))
            loss_r_list.append(loss_r.view(1))

        return torch.cat(loss_add_list, dim=0).mean(), torch.cat(distance_list, dim=0).mean(), torch.cat(loss_r_list, dim=0).mean()
