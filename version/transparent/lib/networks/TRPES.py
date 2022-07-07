#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 19:31
# @Author  : yaomy

# --------------------------------------------------------
# 对之前的网络轻量化了一下， 用Unet作为backbone
# --------------------------------------------------------

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn.functional as F
import torch.nn as nn
from lib.networks.unet import UNet
# from lib.networks.equalized import EqualizedConv1d as Conv1d
# from lib.networks.equalized import EqualizedConv2d as Conv2d
from torch.nn import Conv2d as Conv2d
from torch.nn import Conv1d as Conv1d

from lib.networks.attention import TransformerEncoderLayer


class GeometryNet(nn.Module):
    def __init__(self):
        super(GeometryNet, self).__init__()
        self.conv_1 = Conv2d(192, 64, (1, 1))

    def forward(self, feat_feature, _intrinsic, v, u, _ds):
        bs = feat_feature.size(0)
        ds = _ds.view(bs, 1, 1, 1)
        dx = F.relu(self.conv_1(feat_feature)) * ds
        bs, channel, height_px, width_px = dx.size()

        if _intrinsic.dim() == 3:
            _intrinsic = _intrinsic.view(bs, 3, 3, 1, 1, 1)
            fx, fy, cx, cy = _intrinsic[:, 0, 0], _intrinsic[:, 1, 1], _intrinsic[:, 0, 2], _intrinsic[:, 1, 2]
        else:
            _intrinsic = _intrinsic.view(bs, 4, 1, 1, 1)
            fx, fy, cx, cy = _intrinsic[:, 0], _intrinsic[:, 1], _intrinsic[:, 2], _intrinsic[:, 3]

        u_maps = u.repeat(1, channel, 1, 1)
        v_maps = v.repeat(1, channel, 1, 1)

        pt0 = (u_maps - cx) * dx / fx
        pt1 = (v_maps - cy) * dx / fy

        # (bs, 32, 3, h ,w)
        geometry_feature = torch.stack((pt0, pt1, dx), dim=2)

        return geometry_feature


class Densefusion(nn.Module):
    def __init__(self, num_points):
        super(Densefusion, self).__init__()
        self.conv1 = Conv1d(64, 64, 1)
        self.conv2 = Conv1d(64, 128, 1)

        self.e_conv1_x = Conv1d(64, 64, 1)
        self.e_conv2_x = Conv1d(64, 128, 1)

        self.e_conv1_y = Conv1d(64, 64, 1)
        self.e_conv2_y = Conv1d(64, 128, 1)

        self.e_conv1_z = Conv1d(64, 64, 1)
        self.e_conv2_z = Conv1d(64, 128, 1)

        self.conv5 = Conv1d(512, 1024, 1)

        self.ap1 = nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, geometry_emb, color_emb):
        color_emb = F.relu(self.conv1(color_emb))
        geometry_emb_x = F.relu(self.e_conv1_x(geometry_emb[:, :, 0]))
        geometry_emb_y = F.relu(self.e_conv1_y(geometry_emb[:, :, 1]))
        geometry_emb_z = F.relu(self.e_conv1_z(geometry_emb[:, :, 2]))
        # 256
        pointfeat_1 = torch.cat((color_emb, geometry_emb_x, geometry_emb_y, geometry_emb_z), dim=1)

        color_emb = F.relu(self.conv2(color_emb))
        geometry_emb_x = F.relu(self.e_conv2_x(geometry_emb_x))
        geometry_emb_y = F.relu(self.e_conv2_y(geometry_emb_y))
        geometry_emb_z = F.relu(self.e_conv2_z(geometry_emb_z))
        # 512
        pointfeat_2 = torch.cat((color_emb, geometry_emb_x, geometry_emb_y, geometry_emb_z), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        # 1792 = 1024 + 512 + 256
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)


class PosePredTransformerNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PosePredTransformerNet, self).__init__()
        self.num_obj = num_obj
        self.num_points = num_points

        self.conv1_r = Conv1d(1792, 640, 1)  # quaternion
        self.conv1_t = Conv1d(1792, 640, 1)  # translation
        self.conv1_c = Conv1d(1792, 640, 1)  # confidence

        self.attention2_r = TransformerEncoderLayer(d_model=640, nhead=8)
        self.attention2_t = TransformerEncoderLayer(d_model=640, nhead=4)
        self.attention2_c = TransformerEncoderLayer(d_model=640, nhead=2)

        self.conv3_r = Conv1d(640, 256, 1)  # quaternion
        self.conv3_t = Conv1d(640, 256, 1)  # translation
        self.conv3_c = Conv1d(640, 256, 1)  # confidence

        self.conv3_r = Conv1d(640, 256, 1)  # quaternion
        self.conv3_t = Conv1d(640, 256, 1)  # translation
        self.conv3_c = Conv1d(640, 256, 1)  # confidence

        self.conv4_r = Conv1d(256, self.num_obj * 4, 1)  # quaternion
        self.conv4_t = Conv1d(256, self.num_obj * 3, 1)  # translation
        self.conv4_c = Conv1d(256, self.num_obj * 1, 1)  # confidence

    def forward(self, apx, obj):
        bs = apx.size()[0]
        rx = self.conv1_r(apx)
        tx = self.conv1_t(apx)
        cx = self.conv1_c(apx)

        rx = self.attention2_r(rx.permute((0, 2, 1))).permute((0, 2, 1))
        tx = self.attention2_t(tx.permute((0, 2, 1))).permute((0, 2, 1))
        cx = self.attention2_c(cx.permute((0, 2, 1))).permute((0, 2, 1))

        rx = self.conv3_r(rx)
        tx = self.conv3_t(tx)
        cx = self.conv3_c(cx)

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        # batch select

        out_rx = torch.gather(rx, 1, obj.view(bs, 1, 1, 1).repeat(1, 1, 4, self.num_points)).squeeze(dim=1)
        out_tx = torch.gather(tx, 1, obj.view(bs, 1, 1, 1).repeat(1, 1, 3, self.num_points)).squeeze(dim=1)
        out_cx = torch.gather(cx, 1, obj.view(bs, 1, 1, 1).repeat(1, 1, 1, self.num_points)).squeeze(dim=1)

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx


class PosePred(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PosePred, self).__init__()
        self.num_obj = num_obj
        self.num_points = num_points

        self.conv1_r = Conv1d(1792, 640, 1)  # quaternion
        self.conv1_t = Conv1d(1792, 640, 1)  # translation
        self.conv1_c = Conv1d(1792, 640, 1)  # confidence

        self.conv2_r = Conv1d(640, 256, 1)  # quaternion
        self.conv2_t = Conv1d(640, 256, 1)  # translation
        self.conv2_c = Conv1d(640, 256, 1)  # confidence

        self.conv3_r = Conv1d(256, 128, 1)  # quaternion
        self.conv3_t = Conv1d(256, 128, 1)  # translation
        self.conv3_c = Conv1d(256, 128, 1)  # confidence

        self.conv4_r = Conv1d(128, self.num_obj * 4, 1)  # quaternion
        self.conv4_t = Conv1d(128, self.num_obj * 3, 1)  # translation
        self.conv4_c = Conv1d(128, self.num_obj * 1, 1)  # confidence

    def forward(self, apx, obj):
        bs = apx.size()[0]
        rx = self.conv1_r(apx)
        tx = self.conv1_t(apx)
        cx = self.conv1_c(apx)

        rx = self.conv2_r(rx)
        tx = self.conv2_t(tx)
        cx = self.conv2_c(cx)

        rx = self.conv3_r(rx)
        tx = self.conv3_t(tx)
        cx = self.conv3_c(cx)

        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        # batch select
        out_rx = torch.gather(rx, 1, obj.view(bs, 1, 1, 1).repeat(1, 1, 4, self.num_points)).squeeze(dim=1)
        out_tx = torch.gather(tx, 1, obj.view(bs, 1, 1, 1).repeat(1, 1, 3, self.num_points)).squeeze(dim=1)
        out_cx = torch.gather(cx, 1, obj.view(bs, 1, 1, 1).repeat(1, 1, 1, self.num_points)).squeeze(dim=1)

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        return out_rx, out_tx, out_cx


class TRPESNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(TRPESNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.backbone = UNet()
        self.d2c = GeometryNet()
        self.feat = Densefusion(self.num_points)
        self.pose_predictor = PosePred(self.num_points, self.num_obj)

        self.m_con_1 = nn.Sequential(
            Conv2d(192, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

        self.n_con_1 = nn.Sequential(
            Conv2d(64, 32, kernel_size=(1, 1))
        )
        self.n_con_2 = nn.Sequential(
            Conv2d(32, 3, kernel_size=(1, 1))
        )

        self.d_con_1 = nn.Sequential(
            Conv2d(64, 32, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.d_con_2 = nn.Sequential(
            Conv2d(32, 1, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def forward(self, _img_croped, _intrinsic, _xmap, _ymap, _ds, _obj):
        # [64, h, w]
        p, n, d = self.backbone(_img_croped)
        feat_0 = torch.cat([n, d], dim=1)
        bs, di, h, w = p.size()
        # [64, h, w]
        n = self.n_con_1(n)
        d = self.d_con_1(d)
        # [192, h, w]
        feat_1 = torch.cat([n, d, feat_0], dim=1)
        n = self.n_con_2(n)
        d = self.d_con_2(d)
        # mask
        pred_mask = self.m_con_1(feat_1).contiguous().view(bs, 1, h, w)

        geometry_feature = self.d2c(feat_1, _intrinsic, _xmap, _ymap, _ds)

        _, ddi, xyz, _, _ = geometry_feature.size()
        # choose_list = list()
        # for b in range(bs):
        #     choose_list.append(self.random_sample(pred_mask[0]))
        #
        # choose = torch.cat(choose_list, dim=0)
        choose = torch.randperm(pred_mask[0].numel(), device=pred_mask.device).repeat(bs, 1)[:, :self.num_points].view(
            bs, 1, -1)

        choose_color = choose.repeat(1, di, 1)
        choose_geometry = choose.unsqueeze(dim=1).repeat(1, ddi, xyz, 1).contiguous()

        color_emb = p.contiguous().view(bs, di, -1)
        geometry_emb = geometry_feature.contiguous().view(bs, ddi, xyz, -1)

        color_emb = torch.gather(color_emb, -1, choose_color)
        geometry_emb = torch.gather(geometry_emb, -1, choose_geometry)
        apx = self.feat(geometry_emb, color_emb)
        out_rx, out_tx, out_cx = self.pose_predictor(apx, _obj)
        return out_rx, out_tx, out_cx, n, d, pred_mask

    def random_sample(self, mask):
        positive = torch.where(mask.flatten() >= 0.7)[0]
        if positive.size(-1) < self.num_points:
            positive = torch.where(mask.flatten() >= 0.5)[0]
            if positive.size(-1) < self.num_points:
                positive = F.pad(positive, (self.num_points - positive.size(-1), self.num_points - positive.size(-1)))
        perm = torch.randperm(positive.size(-1), device=positive.device)[:self.num_points]
        return positive[perm].contiguous().view(1, 1, -1)


if __name__ == '__main__':
    # from lib.networks.loss import FocalLoss
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TRPESNet(500, 5)

    model.to(device)
    model.train()
    # criterion_mask = FocalLoss()
    for i in range(10000):
        x = torch.rand(8, 3, 120, 120).to(device)
        m_gt = torch.rand(8, 1, 120, 120).to(device)
        intrinsic = torch.tensor([
            [1386.42, 1386.45, 960.0, 540.0],
        ]).repeat(8, 1).to(device)

        xmap = torch.randn((8, 1, 120, 120)).to(device)
        ymap = torch.randn((8, 1, 120, 120)).to(device)
        bbox = torch.FloatTensor([50., 100., 50., 100.]).unsqueeze(dim=0).to(device)
        _ds = torch.FloatTensor([[1.]]).repeat(8, 1).to(device)

        obj = torch.LongTensor([[0], [0], [1], [1], [2], [2], [3], [3]]).view(8, 1).to(device)
        # print(x.shape, intrinsic.shape, xmap.shape, ymap.shape, _ds, obj.shape)
        r, t, c, n, d, m = model(x, intrinsic, xmap, ymap, _ds, obj)

