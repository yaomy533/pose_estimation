#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 10:13
# @Author  : yaomy

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from mmcv import Config

# from lib.network.hrnet.hhrnet.pose_higher_hrnet import build_modeifed_hhrnet
from lib.network.hrnet.myhrnet  import build_modeifed_hrnet
from lib.network.point.fusion import FusionNet, FusionNetLite
from lib.network.pose.posenet import PoseNet
from lib.network.pose.pose_utils import get_rot_vec_vert_batch, get_rot_mat_y_first
from lib.network.loss_utils import get_xyz


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETCT_PATH = Path(os.path.realpath(__file__)).parent.parent.parent
CONFIG = Config.fromfile(f'{PROJETCT_PATH}/config/linemod/lm_v3_1.py')


class KRRN(nn.Module):
    def __init__(self, num_cls=1, cfg=CONFIG):
        super(KRRN, self).__init__()
        self.cfg = cfg
        self.num_cls = cfg.Module.NUM_CLS

        # 31.73M
        self.backbone = build_modeifed_hrnet(config=cfg, cfg_name=cfg.Module.BACKBONE)
        xyz_channels = self.cfg.Module.XYZNet.HEADEN_FS
        mask_out_channels = self.cfg.Module.MASKNet.OUT_FS * self.num_cls + 1
        xyz_out_channels = self.cfg.Module.XYZNet.OUT_FS * self.num_cls
        region_out_channels = self.cfg.Module.REGIONNet.OUT_FS
        self.mask_outc = mask_out_channels
        self.region_outc = mask_out_channels + region_out_channels
        self.xyz_outc = mask_out_channels + xyz_out_channels + region_out_channels

        nml_channels = self.cfg.Module.NMLNet.HEADEN_FS
        nml_out_channels = self.cfg.Module.NMLNet.OUT_FS * self.num_cls

        # 0.82M
        self.XYZNet = nn.Sequential(
            nn.ConvTranspose2d(cfg.Module.BACKBONE_OUTC, xyz_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1),
                            bias=False),
            nn.BatchNorm2d(xyz_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(xyz_channels, xyz_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(xyz_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2.0),
            nn.Conv2d(xyz_channels, xyz_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(xyz_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(xyz_channels, xyz_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(xyz_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.xyz_final = nn.Conv2d(xyz_channels, mask_out_channels+xyz_out_channels+region_out_channels, kernel_size=(1, 1))


        self.NMLNet = nn.Sequential(
            nn.Conv2d(cfg.Module.BACKBONE_OUTC, nml_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(nml_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            # nn.UpsamplingBilinear2d(scale_factor=2.0),
            nn.Conv2d(nml_channels, nml_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(nml_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2.0),
            nn.Conv2d(nml_channels, nml_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(nml_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.nml_final = nn.Conv2d(nml_channels, nml_out_channels, kernel_size=(1, 1))

        # 5.50M
        # self.fusion = FusionNet(cfg)
        self.fusion = FusionNetLite(cfg)
        self.pose = PoseNet(cfg)

    def forward(self, x, p_emb, choose, cls, region_point=None, opt_pose=True):
        bs = x.size(0)
        xyz_map, nml_map = self.backbone(x)
        xyz_map = self.XYZNet(xyz_map)
        nml_map = self.NMLNet(nml_map)

        xyz_map = self.xyz_final(xyz_map)
        nml_map = self.nml_final(nml_map)

        h, w = xyz_map.size(2), xyz_map.size(3)
        pred_mask = xyz_map[:, 0:self.mask_outc, :, :]
        pred_region = xyz_map[:, self.mask_outc:self.region_outc, :, :]
        xyz_map = xyz_map[:, self.region_outc:self.xyz_outc, :, :]

        xyz_map = torch.gather(xyz_map.view(bs, self.num_cls, 3, h, w), 1, cls.view(bs, 1, 1, 1, 1).repeat(1, 1, 3, h, w)).squeeze(1)
        nml_map = torch.gather(nml_map.view(bs, self.num_cls, 3, h, w), 1, cls.view(bs, 1, 1, 1, 1).repeat(1, 1, 3, h, w)).squeeze(1)

        nml_map = F.normalize(nml_map, p=2, dim=1)

        # 实验12之前的顺序
        # pred_xyz_off = xyz_emb[:, :3, :, :]
        # pred_mask = xyz_emb[:, 3:4, :, :]
        # pred_region = xyz_emb[:, 4:, :, :]

        dx, dn = xyz_map.size(1), nml_map.size(1)
        # pred_xyz = get_xyz(pred_xyz_off, pred_region, region_point)

        pred_xyz = xyz_map
        pred_normal = nml_map
        # xyz_map = region_point
        xyz_emb = torch.gather(xyz_map.contiguous().view(bs, dx, -1), -1, choose.repeat(1, dx, 1)).permute(0, 2, 1)
        nml_emb = torch.gather(nml_map.contiguous().view(bs, dn, -1), -1, choose.repeat(1, dn, 1)).permute(0, 2, 1)

        # num_point = xyz_emb.size(-1)
        # xyz_emb = torch.gather(xyz_emb.view(bs, self.num_cls, 3, num_point), 1, choose.unsqueeze(1).repeat(1, 1, 3, 1))
        # nml_emb = torch.gather(nml_emb.view(bs, self.num_cls, 3, num_point), 1, choose.unsqueeze(1).repeat(1, 1, 3, 1))

        if opt_pose:
            # [8, 1024, 1664]
            feat = self.fusion(p_emb, xyz_emb, nml_emb)

            if cls.shape[0] == 1:
                obj_idh = cls.view(-1, 1).repeat(cls.shape[0], 1)
            else:
                obj_idh = cls.view(-1, 1)

            vertice_num = p_emb.size(1)
            one_hot = torch.zeros(bs, self.num_cls).to(cls.device).scatter_(1, obj_idh.long(), 1)
            one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)
            feat = torch.cat([feat, one_hot], dim=2)

            # feat = torch.randn(8, 1024, 1664)
            # [bs, 4], [bs, 4], [bs, 3]
            rc_red, rc_green, t_res = self.pose(feat.permute(0, 2, 1))

            # r_red = rc_red[:, 1:] / (torch.norm(rc_red[:, 1:], dim=1, keepdim=True) + 1e-6)
            # r_green = rc_green[:, 1:] / (torch.norm(rc_green[:, 1:], dim=1, keepdim=True) + 1e-6)
            # c_green = torch.sigmoid(rc_green[:, 0:1])
            # c_red = torch.sigmoid(rc_red[:, 0:1])
            # new_y, new_x = get_rot_vec_vert_batch(c_red, c_green, r_red, r_green)
            # pred_r = get_rot_mat_y_first(new_y, new_x)
            pred_r = None
            pred_t = (p_emb + t_res.permute(0, 2, 1)).mean(dim=1)
        else:
            pred_r = None
            pred_t = None

        return {
            'xyz': pred_xyz,
            'region': pred_region,
            'mask': pred_mask,
            'normal': pred_normal,
            'pred_r': pred_r,
            'pred_t': pred_t
        }


def main():
    model = KRRN(num_cls=13)
    x = torch.randn(8, 3, 64, 64)
    p = torch.randn(8, 1024, 3)
    choose = torch.randint(0, 1024, (8, 1, 1024)).long()
    cls = torch.randint(0, 12, (8, 1)).long()
    model(x, p, choose, cls)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of total parameter: %.2fM" % (total / 1e6))

    total_bacbone = sum([param.nelement() for param in model.backbone.parameters()])
    total_fusion = sum([param.nelement() for param in model.fusion.parameters()])
    total_pose = sum([param.nelement() for param in model.pose.parameters()])

    print("Number of backbone parameter: %.2fM" % (total_bacbone / 1e6))
    print("Number of decoder parameter: %.2fM" % ((total - total_bacbone - total_fusion - total_pose) / 1e6))
    print("Number of fusion parameter: %.2fM" % (total_fusion / 1e6))
    print("Number of pose parameter: %.2fM" % (total_pose / 1e6))


if __name__ == '__main__':
    main()
