#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/12 15:55
# @Author  : yaomy


import torch
import torch.nn as nn
import torch.nn.functional
from torch.nn import functional as F
from lib.networks.backbone import PsPNet, PSPUpsample

backbone = {
    'pspnet_resnet18': lambda: PsPNet(sizes=(1, 2, 3, 6), psp_size=512, backend='resnet18'),
    'pspnet_resnet34': lambda: PsPNet(sizes=(1, 2, 3, 6), psp_size=512, backend='resnet34')
}


class Backbone(nn.Module):
    def __init__(self, multi_gpu=False):
        super(Backbone, self).__init__()
        self.model = backbone['pspnet_resnet18'.lower()]()

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.drop_1 = nn.Dropout2d(p=0.3)
        self.drop_2 = nn.Dropout2d(p=0.15)

        self.c_up_1 = PSPUpsample(1024, 256)
        self.c_up_2 = PSPUpsample(256, 64)
        self.c_up_3 = PSPUpsample(64, 64)

        self.n_up_1 = PSPUpsample(1024, 256)
        self.n_up_2 = PSPUpsample(256, 64)
        self.n_up_3 = PSPUpsample(64, 64)

        self.d_up_1 = PSPUpsample(1024, 256)
        self.d_up_2 = PSPUpsample(256, 64)
        self.d_up_3 = PSPUpsample(64, 64)

        self.final_c = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 1)),
            nn.LogSoftmax(dim=1)
        )

        self.final_n_1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(1, 1)),
        )

        self.final_n_2 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=(1, 1)),
        )

        self.final_d_1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.final_d_2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.ReLU()
        )

    def forward(self, p):
        p = self.drop_1(p)

        # color
        # [bs, 1024, H / 8, W / 8] -> [bs, 256, H / 4, W / 4]
        c = self.c_up_1(p)
        c = self.drop_2(c)
        # -> [bs, 64, H / 2, W / 2]
        c = self.c_up_2(c)
        c = self.drop_2(c)
        # -> [bs, 64, H, W]
        c = self.c_up_3(c)
        # -> [bs, 32, H, W]
        c = self.final_c(c)

        # normal
        # [bs, 1024, H / 8, W / 8] -> [bs, 256, H /4, W / 4]
        n = self.n_up_1(p.to(torch.float32))
        n = self.drop_2(n)
        # -> [bs, 64, H / 2, W / 2]
        n = self.n_up_2(n)
        n = self.drop_2(n)
        # -> [bs, 64, H, W]
        n_1 = self.n_up_3(n)

        # depth
        # [bs, 1024, H / 8, W / 8] -> [bs, 256, H /4, W / 4]
        d = self.d_up_1(p.to(torch.float32))
        d = self.drop_2(d)
        # -> [bs, 64, H / 2, W / 2]
        d = self.d_up_2(d)
        d = self.drop_2(d)
        # -> [bs, 64, H, W]
        d_1 = self.d_up_3(d)

        # [bs, 64, H, W] + [bs, 64, H, W] -> [bs, 128, H, W]
        f_1 = torch.cat([n_1, d_1], dim=1)
        # [bs, 128, H, W] -> [bs, 32, H, W]
        n_2 = self.final_n_1(f_1)
        d_2 = self.final_d_1(f_1)

        # [bs, 32, H, W] + [bs, 32, H, W] -> [bs, 64, H, W]
        f_2 = torch.cat([n_2, d_2], dim=1)
        # [bs, 64, H, W] -> [bs, 3, H, W]
        n_3 = self.final_n_2(f_2)
        # [bs, 64, H, W] -> [bs, 1, H, W]
        d_3 = self.final_d_2(f_2)

        # [bs, 128, H, W] + [bs, 64, H, W] -> [b    s, 192, H, W]
        f_3 = torch.cat([f_1, f_2], dim=1)

        return c, nn.functional.normalize(n_3), d_3, f_3


class GeoNet(nn.Module):
    def __init__(self):
        super(GeoNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(192, 64, (1, 1))
        self.conv_2 = torch.nn.Conv2d(64, 32, (1, 1))

    def forward(self, feat_feature, _intrinsic, _xmap, _ymap, _ds):
        dx = F.relu(self.conv_1(feat_feature))
        bs = feat_feature.size(0)
        ds = _ds.view(bs, 1, 1, 1)
        dx = F.relu(self.conv_2(dx)) * ds
        bs, channel, height_px, width_px = dx.size()

        if _intrinsic.dim() == 3:
            fx, fy, cx, cy = _intrinsic[0, 0, 0], _intrinsic[0, 1, 1], _intrinsic[0, 0, 2], _intrinsic[0, 1, 2]
        else:
            fx, fy, cx, cy = _intrinsic[0, 0], _intrinsic[0, 1], _intrinsic[0, 2], _intrinsic[0, 3]

        xmap = _xmap.repeat(1, channel, 1, 1)
        ymap = _ymap.repeat(1, channel, 1, 1)
        pt2 = dx

        pt0 = (ymap - cx) * dx / fx
        pt1 = (xmap - cy) * dx / fy

        # (bs, 32, 3, h ,w)
        geometry_feature = torch.stack((pt0, pt1, pt2), dim=2)

        return geometry_feature


class PointFeatNet(nn.Module):
    def __init__(self, num_points):
        super(PointFeatNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(32, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1_x = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2_x = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1_y = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2_y = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1_z = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2_z = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(512, 1024, 1)
        self.conv6 = torch.nn.Conv1d(1024, 2048, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
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
        x = F.relu(self.conv6(x))
        ap_x = self.ap1(x)
        ap_x = ap_x.view(-1, 2048, 1).repeat(1, 1, self.num_points)

        # 2816 = 2048 + 512 + 256
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1)


class PosePred(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PosePred, self).__init__()
        self.num_obj = num_obj
        self.num_points = num_points

        self.conv1_r = nn.Conv1d(2816, 640, 1)  # quaternion
        self.conv1_t = nn.Conv1d(2816, 640, 1)  # translation
        self.conv1_c = nn.Conv1d(2816, 640, 1)  # confidence

        self.conv2_r = nn.Conv1d(640, 256, 1)  # quaternion
        self.conv2_t = nn.Conv1d(640, 256, 1)  # translation
        self.conv2_c = nn.Conv1d(640, 256, 1)  # confidence

        self.conv3_r = nn.Conv1d(256, 128, 1)  # quaternion
        self.conv3_t = nn.Conv1d(256, 128, 1)  # translation
        self.conv3_c = nn.Conv1d(256, 128, 1)  # confidence

        self.conv4_r = nn.Conv1d(128, self.num_obj * 4, 1)  # quaternion
        self.conv4_t = nn.Conv1d(128, self.num_obj * 3, 1)  # translation
        self.conv4_c = nn.Conv1d(128, self.num_obj * 1, 1)  # confidence

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


class BatchPoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(BatchPoseNet, self).__init__()
        self.num_points = num_points
        self.num_obj = num_obj
        self.encoder = Backbone()
        self.decoder = Decoder()
        self.m_con_1 = nn.Sequential(
            nn.Conv2d(192, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )
        # uvd to cloud
        self.d2c = GeoNet()
        self.densefusion = PointFeatNet(self.num_points)
        self.pose_predictor = PosePred(self.num_points, self.num_obj)
        # self.pose_predictor = PosePredTransformerNet(self.num_points, self.num_obj)

    def forward(self, _img_croped, _intrinsic, _xmap, _ymap, _ds, _obj):
        # feature map
        p = self.encoder(_img_croped)
        color_feature, pred_normal, pred_depth, feat_1 = self.decoder(p)

        pred_mask = self.m_con_1(feat_1)
        geometry_feature = self.d2c(feat_1, _intrinsic, _xmap, _ymap, _ds)

        bs, di, h, w = color_feature.size()
        _, ddi, xyz, _, _ = geometry_feature.size()

        # get choose from mask

        # 按照类型随机选点
        # p_m = pred_masks.view(bs, 2, -1).contiguous()
        # foreg_choose = randomly_selected_mask(p_m[:, 0], self.num_points-100, negtive_falg=True)
        # boundry_choose = randomly_selected_mask(p_m[:, 1], 100)
        # choose = torch.cat([foreg_choose, boundry_choose], dim=-1)
        # if choose.size(-1) < self.num_points:
        #     supple_choose = randomly_selected_mask(p_m[:, 0], self.num_points - choose.size(-1))
        #     choose = torch.cat([choose, supple_choose], dim=-1)

        # # 随机选点

        # choose_list = list()
        # for b in range(bs):
        #     choose_list.append(self.random_sample(pred_mask[0]))
        #
        # choose = torch.cat(choose_list, dim=0)
        choose = torch.randperm(pred_mask[0].numel(), device=pred_mask.device).repeat(bs, 1)[:, :self.num_points].view(bs, 1, -1)

        choose_color = choose.repeat(1, di, 1)
        choose_geometry = choose.unsqueeze(dim=1).repeat(1, ddi, xyz, 1).contiguous()

        color_emb = color_feature.contiguous().view(bs, di, -1)
        color_emb = torch.gather(color_emb, -1, choose_color)
        geometry_emb = geometry_feature.contiguous().view(bs, ddi, xyz, -1)
        geometry_emb = torch.gather(geometry_emb, -1, choose_geometry)
        apx = self.densefusion(geometry_emb, color_emb)

        out_rx, out_tx, out_cx = self.pose_predictor(apx, _obj)
        return out_rx, out_tx, out_cx, pred_normal, pred_depth, pred_mask

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
    model = BatchPoseNet(500, 5)

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
        print(r.shape)
