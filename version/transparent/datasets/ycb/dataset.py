#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 18:18
# @Author  : yaomy

import torch.utils.data as data
from PIL import Image
import os
import cv2.cv2 as cv2
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import random
import numpy.ma as ma
from plyfile import PlyData
import copy
import json
from pathlib import Path


class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine, back='/root/Source/ymy_dataset/ms-coco/val2017'):

        self.num_pt = num_pt
        self.root = root
        self.obj_num = 21
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self.all_list = []
        self.backgrounds = []
        self.mode = mode
        self.real = []
        self.syn = []
        self.back = back

        self.gts_dict = self.load_gts()

        # 读取DS整理的部分
        if mode == 'train':
            self.path = f'{self.root}/train_data_list.txt'
            with open(self.path, 'r') as input_file:
                self.all_list = sorted([p.strip() for p in input_file.readlines() if 'syn' in p])
            self.syn = [p for p in self.all_list if 'syn' in p]
            self.real = [p for p in self.all_list if 'syn' not in p]
            # COCO数据集的路径，作为syn的背景
            self.backgrounds = sorted(Path(self.back).glob('*.jpg'))
        elif mode == 'test':# 读取lm整理的部分
            # self.path = f'{self.root}/test_data_list.txt'
            # with open(self.path, 'r') as input_file:
            #     self.all_list = sorted([p.strip() for p in input_file.readlines()])
            self.path = f'{self.root}/test_bop.json'
            with open(self.path, 'r') as input_file:
                al = json.load(input_file)
                self.all_list = [p for p in al if 'im_id' in p.keys()]

        elif mode == 'eval':# 读取bop整理的部分
            self.path = f'{self.root}/test_all.json'
            with open(self.path, 'r') as input_file:
                al = json.load(input_file)
                self.all_list = [p for p in al if 'im_id' in p.keys()]
                # self.all_list = [p for p in self.all_list if (p['scene_id'] in [49, 53]) and (p['obj_id'] == 13)]
                # self.all_list = random.sample(self.all_list, 200)
        else:
            raise KeyError('unkown mode!')

        self.length = len(self.all_list)
        self.test_folder = [k for k in range(48, 60)]
        self.train_folder = [m for m in range(0, 92) if m not in self.test_folder]

        self.height = 480
        self.width = 640

        self.models = {}

        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.refine = refine
        if refine:
            self.pts = load_plys(f'{self.root}/bop/model/models', self.num_pt_mesh_large)
        else:
            self.pts = load_plys(f'{self.root}/bop/model/models', self.num_pt_mesh_small)
        self.front_num = 2
        self.logger = None
        self.ori_iamge = None
        self.info: {}

    def __getitem__(self, index):
        info = self.all_list[index]
        self.info = copy.copy(info)
        source_name = 'train_real'
        datas = copy.copy(self._data)
        if self.mode == 'train':
            path_stem = info[:]
            if 'syn' in info:
                img_id = int(info.split('/')[-1])
                folder_id = img_id // 1000
                source_name = 'train_synt'
                cam_cx, cam_cy, cam_fx, cam_fy = self.cam_cx_1, self.cam_cy_1, self.cam_fx_1, self.cam_fy_1
            else:
                folder_id = int(info.split('/')[1])
                img_id = info.split('/')[-1]
                source_name = 'train_real'
                if folder_id in range(0, 60):
                    cam_cx, cam_cy, cam_fx, cam_fy = self.cam_cx_1, self.cam_cy_1, self.cam_fx_1, self.cam_fy_1
                elif folder_id in range(60, 92):
                    cam_cx, cam_cy, cam_fx, cam_fy = self.cam_cx_2, self.cam_cy_2, self.cam_fx_2, self.cam_fy_2
                else:
                    raise KeyError('no this folder!')
        elif self.mode == 'test':
            info: {}
            folder_id = info['scene_id']
            img_id = info['im_id']
            source_name = 'test_part'
            path_stem = f'data/{folder_id:04d}/{img_id:06d}'
            cam_cx, cam_cy, cam_fx, cam_fy = self.cam_cx_1, self.cam_cy_1, self.cam_fx_1, self.cam_fy_1
        elif self.mode == 'eval':
            info: {}
            folder_id = info['scene_id']
            img_id = info['im_id']
            source_name = 'test_all'
            path_stem = f'data/{folder_id:04d}/{img_id:06d}'
            cam_cx, cam_cy, cam_fx, cam_fy = self.cam_cx_1, self.cam_cy_1, self.cam_fx_1, self.cam_fy_1
        else:
            raise KeyError('unkown mode!')

        img_path = f'{self.root}/bop/{source_name}/{folder_id:06d}/rgb/{int(img_id):06d}.png'
        depth_path = f'{self.root}/bop/{source_name}/{folder_id:06d}/depth/{int(img_id):06d}.png'
        normal_path = f'{self.root}/bop/{source_name}/{folder_id:06d}/normal/{int(img_id):06d}.png'

        img = Image.open(img_path).convert("RGB")
        self.ori_img = np.array(img)
        depth = np.asarray(Image.open(depth_path))

        normal = cv2.imread(normal_path, -1)
        gts = self.gts_dict[source_name][f'{folder_id:06d}'][str(int(img_id))]

        if self.mode == 'train' or self.mode == 'test':
            idx = np.random.randint(0, len(gts))
        elif self.mode == 'eval':
            for id, g in enumerate(gts):
                info: {}
                if g['obj_id'] == info['obj_id']:
                    idx = id
                    break
        else:
            raise KeyError('unkown mode!')

        label_path = f'{self.root}/{path_stem}-label.png'
        label = np.array(Image.open(label_path))

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))
        gt = gts[idx]
        target_r = np.array(gt['cam_R_m2c']).reshape((3, 3))
        target_t = np.array(gt['cam_t_m2c']).reshape((1, 3))/1000.0
        obj_id = gt['obj_id']

        # 从syn中随机选择物体来遮挡
        add_front = False
        if self.add_noise:
            for k in range(5):
                seed = random.choice(self.syn)
                front = np.array(self.trancolor(Image.open('{0}/{1}-color.png'.format(self.root, seed)).convert("RGB")))
                front = np.transpose(front, (2, 0, 1))
                f_label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, seed)))
                front_label = np.unique(f_label).tolist()[1:]
                if len(front_label) < self.front_num:
                   continue
                front_label = random.sample(front_label, self.front_num)
                for f_i in front_label:
                    mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
                    if f_i == front_label[0]:
                        mask_front = mk
                    else:
                        mask_front = mask_front * mk

                t_label = label * mask_front
                if len(t_label.nonzero()[0]) > 1000:
                    label = t_label
                    add_front = True
                    break

        mask_label = ma.getmaskarray(ma.masked_equal(label, obj_id))
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask = mask_label * mask_depth
        if len(mask.nonzero()[0]) < self.minimum_num_pt:
            return self._error_data

        if self.add_noise:
            img = np.array(self.trancolor(img))

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        mask_croped = mask_label[rmin:rmax, cmin:cmax].astype(np.uint8)
        contours, _ = cv2.findContours(mask_croped * 255, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        u = np.squeeze(contours[0][:, :, 0])
        v = np.squeeze(contours[0][:, :, 1])
        mask_con = np.zeros(mask_croped.shape)
        mask_con[v, u] = 1

        datas['masks'].append(
            torch.from_numpy(mask_croped.astype(np.float32)).unsqueeze(dim=0)
        )

        datas['boundary'].append(
            torch.from_numpy(mask_con.astype(np.float32)).unsqueeze(dim=0)
        )

        datas['xmaps'].append(torch.from_numpy(self.xmap[rmin:rmax, cmin:cmax].astype(np.float32)).unsqueeze(dim=0))
        datas['ymaps'].append(torch.from_numpy(self.ymap[rmin:rmax, cmin:cmax].astype(np.float32)).unsqueeze(dim=0))

        img = np.transpose(img, (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        # 从coco val中选择随机背景作为syn的背景
        if 'data_syn' in info:
            seed = random.choice(self.backgrounds)
            # background = cv2.resize(np.array(Image.open(seed)), (self.width, self.height))
            background = Image.open(seed).convert("RGB")
            back = cv2.resize(np.array(self.trancolor(background)), (self.width, self.height))
            back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        else:
            img_masked = img

        if self.add_noise and add_front:
            img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] \
                         + front[:, rmin:rmax, cmin:cmax] * ~(mask_front[rmin:rmax, cmin:cmax])

        datas['img_cropeds'].append(self.norm(torch.from_numpy(img_masked.astype(np.float32))))
        model_points = self.pts[obj_id].copy()
        datas['model_points'].append(torch.from_numpy(model_points.astype(np.float32)))

        target = np.dot(model_points, target_r.T) + target_t

        datas['targets'].append(torch.from_numpy(target.astype(np.float32)))
        datas['obj_ids'].append(torch.IntTensor([int(obj_id)-1]))

        depth_masked = depth[rmin:rmax, cmin:cmax]
        cam_scale = 10000.0
        # d_scale = cam_scale * max([rmax-rmin, cmax-cmin]) / self.diameters[objid]
        d_scale = cam_scale * (rmax - rmin) * (cmax - cmin) / (self.height * self.width)
        depth_nromalized = depth_masked / d_scale

        datas['d_scales'].append(torch.FloatTensor([d_scale]))
        datas['depths'].append(torch.from_numpy(depth_nromalized.astype(np.float32)).unsqueeze(dim=0))

        normal_masked = normal[rmin:rmax, cmin:cmax, :].astype(np.float32).transpose((2, 0, 1))
        normal_masked = F.normalize(torch.from_numpy(normal_masked).double(), p=2, dim=0)

        datas['normals'].append(normal_masked)

        datas['intrinsic'].append(torch.FloatTensor([cam_fx, cam_fy, cam_cx, cam_cy]))
        datas['bboxes'].append(torch.FloatTensor([rmin, rmax, cmin, cmax]))
        datas['axis'].append(torch.from_numpy(np.array([0., 0., 0.]).astype(np.float32)))
        datas['target_rs'].append(torch.from_numpy(target_r.astype(np.float32)))
        datas['target_ts'].append(torch.from_numpy(target_t.astype(np.float32)))

        return datas

    def load_gts(self):
        print('Read all annotations...')
        gts = {}
        i = 0
        if self.mode == 'train':
            types = ['train_real', 'train_synt']
        elif self.mode == 'test':
            types = ['test_part']
        elif self.mode == 'eval':
            types = ['test_all']
        else:
            raise KeyError('unkown mode!')

        for tp in types:
            type_gts = {}
            folder_root = os.path.join(f'{self.root}/bop/{tp}')
            folders = os.listdir(folder_root)
            for folder in folders:
                i += 1
                gt = self.load_json(os.path.join(folder_root, folder, 'scene_gt.json'))
                type_gts[folder] = copy.copy(gt)
                del gt
            gts[tp] = type_gts
        print(f'{i} label files were successfully read.')
        return gts

    @staticmethod
    def load_json(path):
        with open(path) as f_json:
            gts = json.load(f_json)
        return gts

    @property
    def _error_data(self):
        cc = torch.FloatTensor([0.])
        error_datas = {
            'img_cropeds': cc,
            'intrinsic': cc,
            'depths': cc,
            'bboxes': cc,
            'obj_ids': cc,
            'targets': cc,
            'd_scales': cc,
            'model_points': cc,
            'masks': cc,
            'normals': cc,
            'flag': False
        }
        return error_datas

    @property
    def _data(self):
        datas = {
            'img_cropeds': [],
            'intrinsic': [],
            'depths': [],
            'bboxes': [],
            'obj_ids': [],
            'targets': [],
            'd_scales': [],
            'model_points': [],
            'masks': [],
            'normals': [],
            'corners': [],
            'axis': [],
            'target_rs': [],
            'target_ts': [],
            'xmaps': [],
            'ymaps': [],
            'boundary': [],
            'sourec': '',
            'flag': True,
        }
        return datas

    def __len__(self):
        return len(self.all_list)

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    @property
    def suc_dict(self):
        return {str(i): 0 for i in range(self.obj_num)}

    @property
    def num_dict(self):
        return {str(i): 0 for i in range(self.obj_num)}

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_index_from_path(self, path):
        return self._datalist_input.index(path)

    def get_diameter(self):
        return self.diameters

    @property
    def res_name(self):
        return {
            '0': 'master_chef_can',
            '1': 'cracker_box',
            '2': 'sugar_box',
            '3': 'tomato_soup_can',
            '4': 'mustard_bottle',
            '5': 'tuna_fish_can',
            '6': 'pudding_box',
            '7': 'gelatin_box',
            '8': 'potted_meat_can',
            '9': 'banana',
            '10': 'pitcher_base',
            '11': 'bleach_cleanser',
            '12': 'bowl',
            '13': 'mug',
            '14': 'power_drill',
            '15': 'wood_block',
            '16': 'scissors',
            '17': 'large_marker',
            '18': 'large_clamp',
            '19': 'extra_large_clamp',
            '20': 'foam_brick',
        }


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    xlist = plydata['vertex']['x']
    ylist = plydata['vertex']['y']
    zlist = plydata['vertex']['z']
    pc_array = np.array([[x, y, z] for x, y, z in zip(xlist, ylist, zlist)])
    return pc_array


def load_plys(model_file, num):
    """ read 21 models in ycbv and sample n points """
    pts = {}
    for i in range(1, 22):
        model_path = f'{model_file}/obj_{i:06d}.ply'
        points = read_ply(model_path)/1000.0
        dellist = [j for j in range(0, len(points))]
        dellist = random.sample(dellist, len(points) - num)
        pts[i] = np.delete(points, dellist, axis=0)
    print('Read 21 object models in ycbv')
    return pts


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
width_px = img_width = 640
height_px = img_length = 480


def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if border_list[tt] < r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if border_list[tt] < c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_length:
        delt = rmax - img_length
        rmax = img_length
        rmin -= delt

    if cmax > img_width:
        delt = cmax - img_width
        cmax = img_width
        cmin -= delt
    return rmin, rmax, cmin, cmax


def get_square_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin  # h
    c_b = cmax - cmin  # w
    if r_b <= c_b:
        r_b = c_b
    else:
        c_b = r_b
    for tt in range(len(border_list)):
        if border_list[tt] < r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break

    for tt in range(len(border_list)):
        if border_list[tt] < c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > height_px:
        delt = rmax - height_px
        rmax = height_px
        rmin -= delt
        if rmin < 0:
            rmax = rmax - rmin
            rmin = 0
            if rmax >= height_px:
                rmax = height_px - 1

    if cmax > width_px:
        delt = cmax - width_px
        cmax = width_px
        cmin -= delt
        if cmin < 0:
            cmax = cmax - cmin
            cmin = 0
            if cmax >= width_px:
                cmax = width_px - 1
    m = (rmax - rmin) - (cmax - cmin)
    if m > 0:
        rmax = rmax - np.floor(m / 2)
        rmin = rmin + np.floor(m / 2)
    elif m < 0:
        cmax = cmax + np.floor(m / 2)
        cmin = cmin - np.floor(m / 2)

    return int(rmin), int(rmax), int(cmin), int(cmax)


if __name__ == '__main__':
    dataset = PoseDataset('test', 1000, False, '/root/Source/ymy_dataset/YCB-V', 0.03, False)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1, num_workers=0)
    data_iters = dataloader.__iter__()
    while True:
        while True:
            ds = data_iters.__next__()
            if ds['flag']:
                break

        ori_img = dataset.ori_img
        # img_croped = datas['img_cropeds'][0].squeeze().permute((1, 2, 0))
        bbox = ds['bboxes'][0].squeeze().int()
        img_croped = ori_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        mask = ds['masks'][0].squeeze()
        depth = ds['depths'][0].squeeze()
        normal = ds['normals'][0].squeeze().permute((1, 2, 0))
        import matplotlib.pyplot as plt
        plt.subplot(221)
        plt.axis('off')
        plt.imshow(img_croped)
        plt.subplot(222)
        plt.axis('off')
        plt.imshow(mask)
        plt.subplot(223)
        plt.axis('off')
        plt.imshow(depth)
        plt.subplot(224)
        plt.axis('off')
        plt.imshow(normal)
        plt.show()

        while True:
            if cv2.waitKey():
                break
