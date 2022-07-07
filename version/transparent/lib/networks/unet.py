#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/17 18:47
# @Author  : yaomy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, relu=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if relu:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, relu=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, relu)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels, relu)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = 3
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.up1_nd = Up(1024, 512 // factor, bilinear)
        self.up2_nd = Up(512, 256 // factor, bilinear)

        self.up3_n = Up(256, 128 // factor, bilinear)
        self.up4_n = Up(128, 64, bilinear)

        self.up3_d = Up(256, 128 // factor, bilinear)
        self.up4_d = Up(128, 64, bilinear)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_nd = self.up1(x5, x4)
        x_nd = self.up2(x_nd, x3)

        x_n = self.up3(x_nd, x2)
        x_n = self.up4(x_n, x1)

        x_d = self.up3(x_nd, x2)
        x_d = self.up4(x_d, x1)

        return x, nn.functional.normalize(x_n), x_d


if __name__ == '__main__':
    # model = UNet()
    # model.cuda()
    # data = torch.randn((8, 3, 120, 120)).cuda()
    # feature = model(data)
    # print(feature.shape)

    # from lib.networks.attention import TransformerEncoderLayer
    # encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
    # src = torch.rand((10, 32, 512))
    # out = encoder_layer(src)
    # print(out.shape)

    m = nn.Linear(20, 30)
    input = torch.randn(128, 20)
    output = m(input)
