#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 10:29
# @Author  : yaomy
import torch
import torch.nn as nn
import torch.nn.functional as F
import lib.network.point.gcn3d as gcn3d

import os
from pathlib import Path
from mmcv import Config

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETCT_PATH = Path(os.path.realpath(__file__)).parent.parent.parent.parent
CONFIG = Config.fromfile(f'{PROJETCT_PATH}/config/linemod/lm_v3.py')


class FusionNet(nn.Module):
    def __init__(self, cfg=CONFIG):
        super(FusionNet, self).__init__()
        # 10
        self.neighbor_num = cfg.Module.GCN3D.GCN_N_NUM
        # 7
        self.support_num = cfg.Module.GCN3D.GCN_SUP_NUM
        self.num_cls = cfg.Module.NUM_CLS

        self.conv_0_v = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1_v = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1_v = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2_v = gcn3d.Conv_layer(128, 256, support_num=self.support_num)
        self.conv_3_v = gcn3d.Conv_layer(256, 256, support_num=self.support_num)
        self.bn1_v = nn.BatchNorm1d(128)
        self.bn2_v = nn.BatchNorm1d(256)
        self.bn3_v = nn.BatchNorm1d(256)

        self.conv_0_x = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1_x = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1_x = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2_x = gcn3d.Conv_layer(128, 256, support_num=self.support_num)
        self.conv_3_x = gcn3d.Conv_layer(256, 256, support_num=self.support_num)
        self.bn1_x = nn.BatchNorm1d(128)
        self.bn2_x = nn.BatchNorm1d(256)
        self.bn3_x = nn.BatchNorm1d(256)

        self.conv_0_n = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1_n = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1_n = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2_n = gcn3d.Conv_layer(128, 256, support_num=self.support_num)
        self.conv_3_n = gcn3d.Conv_layer(256, 256, support_num=self.support_num)
        self.bn1_n = nn.BatchNorm1d(128)
        self.bn2_n = nn.BatchNorm1d(256)
        self.bn3_n = nn.BatchNorm1d(256)

        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.Conv_fuse_layer(768, 256, support_num=self.support_num)
        self.conv_5 = gcn3d.Conv_fuse_layer(256, 512, support_num=self.support_num)

    def forward(self, vertices, xyz, normal):

        # [bs, N, 10]
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        # [bs, N, 128]
        fm_0_v = F.relu(self.conv_0_v(neighbor_index, vertices), inplace=True)
        fm_0_x = F.relu(self.conv_0_x(neighbor_index, xyz), inplace=True)
        fm_0_n = F.relu(self.conv_0_n(neighbor_index, normal), inplace=True)

        # [bs, N, 128]
        fm_1_v = F.relu(self.bn1_v(self.conv_1_v(neighbor_index, vertices, fm_0_v).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_1_x = F.relu(self.bn1_x(self.conv_1_x(neighbor_index, xyz, fm_0_x).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_1_n = F.relu(self.bn1_n(self.conv_1_n(neighbor_index, normal, fm_0_n).transpose(1, 2)).transpose(1, 2),
                      inplace=True)

        # [bs, N/4, 384]
        feat_1 = torch.cat([fm_1_v, fm_1_x, fm_1_n], 2)

        # [bs, N/4, 9]
        feat_feature = torch.cat([vertices, xyz, normal], 2)

        # v_pool_1:[bs, N/4, 3], fm_pool_1_v:[bs, N/4, 128]
        v_pool_1, fm_pool_1_v = self.pool_1_v(vertices, fm_1_v)
        x_pool_1, fm_pool_1_x = self.pool_1_x(xyz, fm_1_x)
        n_pool_1, fm_pool_1_n = self.pool_1_n(normal, fm_1_n)

        # pool_1: [bs, N/4, 9]  fm_pool: [bs, N/4, 384]
        pool_1, fm_pool_1 = self.pool_1(feat_feature, feat_1)

        # [bs, N/4, 10]
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1, min(self.neighbor_num, v_pool_1.shape[1] // 8))

        # [bs, N/4, 256]
        fm_2_v = F.relu(self.bn2_v(self.conv_2_v(neighbor_index, v_pool_1, fm_pool_1_v).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_2_x = F.relu(self.bn2_x(self.conv_2_x(neighbor_index, x_pool_1, fm_pool_1_x).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_2_n = F.relu(self.bn2_n(self.conv_2_n(neighbor_index, n_pool_1, fm_pool_1_n).transpose(1, 2)).transpose(1, 2),
                      inplace=True)

        # [bs, N/4, 256]
        fm_3_v = F.relu(self.bn3_v(self.conv_3_v(neighbor_index, v_pool_1, fm_2_v).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_3_x = F.relu(self.bn3_x(self.conv_3_x(neighbor_index, x_pool_1, fm_2_x).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_3_n = F.relu(self.bn3_n(self.conv_3_n(neighbor_index, n_pool_1, fm_2_n).transpose(1, 2)).transpose(1, 2),
                      inplace=True)

        # [bs, N/4, 768]
        feat_2 = torch.cat([fm_3_v, fm_3_x, fm_3_n], 2)

        # pool_2:[bs, N/16, 9]  fm_pool_2: [bs, N/16, 768]
        pool_2, fm_pool_2 = self.pool_2(pool_1, feat_2)


        # [bs, N/16, 10]
        neighbor_index = gcn3d.get_neighbor_index(pool_2, min(self.neighbor_num,
                                                                pool_2.shape[1] // 8))

        # [bs, N/16, 512]
        fm_4 = self.conv_4(neighbor_index, pool_2, fm_pool_2)
        # [bs, N/16, 512]
        fm_5 = self.conv_5(neighbor_index, pool_2, fm_4)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, pool_1[..., :3])
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, pool_2[..., :3])

        feat_1 = gcn3d.indexing_neighbor(feat_1, nearest_pool_1).squeeze(2)
        feat_2 = gcn3d.indexing_neighbor(feat_2, nearest_pool_1).squeeze(2)
        fm_5 = gcn3d.indexing_neighbor(fm_5, nearest_pool_2).squeeze(2)
        feat = torch.cat([fm_5, feat_1, feat_2], dim=2)

        return feat

class FusionNetLite(nn.Module):
    def __init__(self, cfg=CONFIG):
        super(FusionNetLite, self).__init__()
        # 10
        self.neighbor_num = cfg.Module.GCN3D.GCN_N_NUM
        # 7
        self.support_num = cfg.Module.GCN3D.GCN_SUP_NUM
        self.num_cls = cfg.Module.NUM_CLS

        self.conv_0_v = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1_v = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1_v = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2_v = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.bn1_v = nn.BatchNorm1d(128)
        self.bn2_v = nn.BatchNorm1d(128)

        self.conv_0_x = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1_x = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1_x = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2_x = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.bn1_x = nn.BatchNorm1d(128)
        self.bn2_x = nn.BatchNorm1d(128)

        self.conv_0_n = gcn3d.Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1_n = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1_n = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2_n = gcn3d.Conv_layer(128, 128, support_num=self.support_num)
        self.bn1_n = nn.BatchNorm1d(128)
        self.bn2_n = nn.BatchNorm1d(128)

        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.Conv_fuse_layer(384, 512, support_num=self.support_num)
        self.conv_5 = gcn3d.Conv_fuse_layer(512, 512, support_num=self.support_num)

    def forward(self, vertices, xyz, normal):

        # [bs, N, 10]
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        # [bs, N, 128]
        fm_0_v = F.relu(self.conv_0_v(neighbor_index, vertices), inplace=True)
        fm_0_x = F.relu(self.conv_0_x(neighbor_index, xyz), inplace=True)
        fm_0_n = F.relu(self.conv_0_n(neighbor_index, normal), inplace=True)

        # [bs, N, 128]
        fm_1_v = F.relu(self.bn1_v(self.conv_1_v(neighbor_index, vertices, fm_0_v).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_1_x = F.relu(self.bn1_x(self.conv_1_x(neighbor_index, xyz, fm_0_x).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_1_n = F.relu(self.bn1_n(self.conv_1_n(neighbor_index, normal, fm_0_n).transpose(1, 2)).transpose(1, 2),
                      inplace=True)

        # [bs, N/4, 384]
        feat_1 = torch.cat([fm_1_v, fm_1_x, fm_1_n], 2)

        # [bs, N/4, 9]
        feat_feature = torch.cat([vertices, xyz, normal], 2)

        # v_pool_1:[bs, N/4, 3], fm_pool_1_v:[bs, N/4, 128]
        v_pool_1, fm_pool_1_v = self.pool_1_v(vertices, fm_1_v)
        x_pool_1, fm_pool_1_x = self.pool_1_x(xyz, fm_1_x)
        n_pool_1, fm_pool_1_n = self.pool_1_n(normal, fm_1_n)

        # pool_1: [bs, N/4, 9]  fm_pool: [bs, N/4, 384]
        pool_1, fm_pool_1 = self.pool_1(feat_feature, feat_1)

        # [bs, N/4, 10]
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1.detach(), min(self.neighbor_num, v_pool_1.shape[1] // 8))

        # [bs, N/4, 256]
        fm_2_v = F.relu(self.bn2_v(self.conv_2_v(neighbor_index, v_pool_1, fm_pool_1_v).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_2_x = F.relu(self.bn2_x(self.conv_2_x(neighbor_index, x_pool_1, fm_pool_1_x).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_2_n = F.relu(self.bn2_n(self.conv_2_n(neighbor_index, n_pool_1, fm_pool_1_n).transpose(1, 2)).transpose(1, 2),
                      inplace=True)

        # [bs, N/4, 768]
        feat_2 = torch.cat([fm_2_v, fm_2_x, fm_2_n], 2)

        # pool_2:[bs, N/16, 9]  fm_pool_2: [bs, N/16, 768]
        pool_2, fm_pool_2 = self.pool_2(pool_1, feat_2)


        # [bs, N/16, 10]
        neighbor_index = gcn3d.get_neighbor_index(pool_2.detach(), min(self.neighbor_num,
                                                                pool_2.shape[1] // 8))

        # [bs, N/16, 512]
        fm_4 = self.conv_4(neighbor_index, pool_2, fm_pool_2)
        # [bs, N/16, 512]
        fm_5 = self.conv_5(neighbor_index, pool_2, fm_4)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices.detach(), pool_1[..., :3].detach())
        nearest_pool_2 = gcn3d.get_nearest_index(vertices.detach(), pool_2[..., :3].detach())

        feat_1 = gcn3d.indexing_neighbor(feat_1, nearest_pool_1).squeeze(2)
        feat_2 = gcn3d.indexing_neighbor(feat_2, nearest_pool_1).squeeze(2)
        fm_5 = gcn3d.indexing_neighbor(fm_5, nearest_pool_2).squeeze(2)
        # 1280 = 128*3+128*3+512
        feat = torch.cat([fm_5, feat_1, feat_2], dim=2)

        return feat

def main():
    vertices = torch.randn(8, 1024, 3)
    xyz = torch.randn(8, 1024, 3)
    normal = torch.randn(8, 1024, 3)
    model = FusionNet()
    out = model(vertices, xyz, normal)
    print(out.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


if __name__ == '__main__':
    main()
