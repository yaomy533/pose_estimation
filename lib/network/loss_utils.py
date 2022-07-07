#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 20:00
# @Author  : yaomy
import torch
import torch.nn as nn

def cosine(x, tgt):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    return 1.0 - cos(x, tgt)

def l1(x, tgt):
    return torch.abs(x-tgt).sum(dim=1)

def cross_entropy(x, tgt, _EPS=1e-6):
    x = -torch.log(torch.softmax(x, 1) + torch.tensor(_EPS))
    return torch.gather(x, 1, tgt).squeeze(1)

def get_xyz(xyz_off: torch.Tensor, region: torch.Tensor, region_point: torch.Tensor):
    """
    :param xyz_off: [bs, 3, h, w]
    :param region: [bs, N, h, w] N is region num
    :param region_point: [bs, N, 3]
    """
    region = torch.softmax(region, 1)
    bs, c, h, w = xyz_off.size()
    n = region.size(1)
    base = xyz_off + (region.unsqueeze(2) * region_point.view(bs, n, 3, 1, 1)).mean(dim=1)
    return base

# def loss_map(x, target, fn=cosine, reduction='elementwise_mean'):
#     loss = fn(x, target)
#     mask_invalid_pixels = torch.all(target == 0., dim=1)
#     loss[mask_invalid_pixels] = 0.0
#     loss_cos_sum = loss.sum()
#     total_valid_pixels = (~mask_invalid_pixels).sum().double()
#     error_output = loss_cos_sum / total_valid_pixels
#
#     if reduction == 'elementwise_mean':
#         final_loss = error_output
#     elif reduction == 'sum':
#         final_loss = loss_cos_sum
#     elif reduction == 'none':
#         final_loss = loss
#     else:
#         raise Exception(
#             'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())
#
#     return final_loss


class MapLoss(nn.Module):
    def __init__(self, fn, reduction='elementwise_mean'):
        super(MapLoss, self).__init__()
        self.function = fn
        self.reduction = reduction
    def forward(self, x, target):
        loss = self.function(x, target)
        mask_invalid_pixels = torch.all(target == 0., dim=1)
        loss[mask_invalid_pixels] = 0.0
        loss_cos_sum = loss.sum()
        total_valid_pixels = (~mask_invalid_pixels).sum().double()
        error_output = loss_cos_sum / total_valid_pixels

        if self.reduction == 'elementwise_mean':
            final_loss = error_output
        elif self.reduction == 'sum':
            final_loss = loss_cos_sum
        elif self.reduction == 'none':
            final_loss = loss
        else:
            raise Exception(
                'Invalid value for reduction  parameter passed. Please use \'elementwise_mean\' or \'none\''.format())
        return final_loss


def main():
    xyz_off = torch.randn(8, 3, 128, 128)
    region = torch.randn(8, 64, 128, 128)
    region_point = torch.randn(8, 64, 3)
    target = torch.randn(8, 3, 128, 128)
    print(region.argmax(1).shape)

if __name__ == '__main__':
    main()