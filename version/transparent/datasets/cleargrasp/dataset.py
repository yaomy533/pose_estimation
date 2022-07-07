#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/10/27 21:04
# @Author  : yaomy

import os
import glob
from PIL import Image
import numpy as np
import yaml
import copy
import plyfile
import json
# import imgaug as ia
import cv2.cv2 as cv2
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn import functional
# from imgaug import augmenters as iaa
from attrdict import AttrDict
from lib.utils import sample_points_from_mesh, pairwise_distance
from lib.transform.rotation import rotation_axis

from numpy import ma
import matplotlib.pyplot as plt
import os

CONFIG_FILE_PATH = f'{os.path.dirname(__file__)}/dataconfig/config.yaml'

with open(CONFIG_FILE_PATH) as fd:
    config_yaml = yaml.load(fd, Loader=yaml.FullLoader)
CONFIGS = AttrDict(config_yaml)


class PoseDataset(Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine, config=None, max_img=560):
        if config is None:
            config = CONFIGS
        self.refine = refine
        self.num_pt_mesh_small = 1000
        self.num_pt_mesh_large = 2600
        self.config = config
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.all_list = []
        self.mode = mode
        self._init_paramter()
        # 读取数据
        self._create_lists_filenames(self.images_dir, self.gt_normals_dir, self.mask_dir, self.depth_dir, self.json_dir)
        self.pts = get_model(self.root, self.num_pt)
        self.xmap = np.array([[j for i in range(self.width_px)] for j in range(self.height_px)])
        self.ymap = np.array([[i for i in range(self.width_px)] for j in range(self.height_px)])
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        # self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.diameters = [0.127, 0.0942, 0.0632, 0.1726, 0.155]
        # self.diameters = [0.1419, 0.1294, 0.0908, 0.1998, 0.1772]
        self.ori_img = np.zeros((1080, 1920, 3))
        self.obj_num = 5
        self.max_img = max_img

    def _create_lists_filenames(self, images_dir, gt_normals_dir, masks_dir, depth_dir, json_dir):
        for ext in self._extension_input:
            imageSearchStr = os.path.join(self.root, images_dir, '*' + ext)
            imagepaths = sorted(glob.glob(imageSearchStr))
            self._datalist_input = self._datalist_input + imagepaths

        numImages = len(self._datalist_input)
        if numImages == 0:
            raise ValueError('No images found in given directory. Searched in dir: {} '.format(images_dir))

        if gt_normals_dir:
            for ext in self._extension_gt_normals:
                gt_normalsSearchStr = os.path.join(self.root, gt_normals_dir, '*' + ext)
                gt_normalspaths = sorted(glob.glob(gt_normalsSearchStr))
                self._datalist_gt_normals = self._datalist_gt_normals + gt_normalspaths

            numgt_normals = len(self._datalist_gt_normals)
            if numgt_normals == 0:
                raise ValueError('No gt_normals found in given directory. Searched for {}'.format(gt_normals_dir))
            if numImages != numgt_normals:
                raise ValueError('The number of images and gt_normals do not match. Please check data,' +
                                 'found {} images and {} gt_normals in dirs:\n'.format(numImages, numgt_normals) +
                                 'images: {}\ngt_normals: {}\n'.format(images_dir, gt_normals_dir))
        if depth_dir:
            for ext in self._extension_depth:
                depthSearchStr = os.path.join(self.root, depth_dir, '*' + ext)

                depthpaths = sorted(glob.glob(depthSearchStr))
                self._datalist_depth = self._datalist_depth + depthpaths

            numdepth = len(self._datalist_depth)
            if numdepth == 0:
                raise ValueError('No depths found in given directory. Searched for {}'.format(imageSearchStr))
            if numImages != numdepth:
                raise ValueError('The number of images and depth do not match. Please check data,' +
                                 'found {} images and {} depth in dirs:\n'.format(numImages, numdepth) +
                                 'images: {}\ndepth: {}\n'.format(images_dir, depth_dir))

        if json_dir:
            for ext in self._extension_json:
                jsonSearchStr = os.path.join(self.root, json_dir, '*' + ext)
                jsonpaths = sorted(glob.glob(jsonSearchStr))
                self._datalist_json = self._datalist_json + jsonpaths

            numjson = len(self._datalist_json)
            if numjson == 0:
                raise ValueError('No json found in given directory. Searched for {}'.format(jsonSearchStr))
            if numImages != numjson:
                raise ValueError('The number of images and json do not match. Please check data,' +
                                 'found {} images and {} json in dirs:\n'.format(numImages, numjson) +
                                 'images: {}\njson: {}\n'.format(images_dir, json_dir))

        if masks_dir:
            for ext in self._extension_mask:
                maskSearchStr = os.path.join(self.root, masks_dir, '*' + ext)
                maskpaths = sorted(glob.glob(maskSearchStr))
                self._datalist_mask = self._datalist_mask + maskpaths

            numMasks = len(self._datalist_mask)
            if numMasks == 0:
                raise ValueError('No masks found in given directory. Searched for {}'.format(imageSearchStr))
            if numImages != numMasks:
                raise ValueError('The number of images and masks do not match. Please check data,' +
                                 'found {} images and {} masks in dirs:\n'.format(numImages, numMasks) +
                                 'images: {}\nmasks: {}\n'.format(images_dir, masks_dir))

    def _init_paramter(self):
        # 原始的不加resize
        self.height_px = 1080
        self.width_px = 1920

        self.height_px_2 = 576
        self.width_px_2 = 1024

        self.xmap = np.array([[j for _ in range(self.width_px)] for j in range(self.height_px)])
        self.ymap = np.array([[i for i in range(self.width_px)] for _ in range(self.height_px)])

        self.xmap_2 = np.array([[j for _ in range(self.width_px_2)] for j in range(self.height_px_2)])
        self.ymap_2 = np.array([[i for i in range(self.width_px_2)] for _ in range(self.height_px_2)])

        self._extension_input = ['-rgb.jpg']  # The file extension of input images
        self._extension_gt_normals = ['-cameraNormals.exr']
        self._extension_depth = ['-depth-rectified.exr']
        self._extension_mask = ['-variantMasks.exr']
        self._extension_json = ['-masks.json']
        self._extension_plane = ['-plane.exr']
        self._datalist_input = []  # Variable containing list of all input images filenames in dataset
        self._datalist_gt_normals = []
        self._datalist_depth = []
        self._datalist_mask = []
        self._datalist_json = []
        self._datalist_plane = []
        self.symmetry_obj_idx = [0, 1, 2, 3, 4] # 全部都对称
        if self.mode == 'train':
            self.images_dir = self.config.train.datasetsTrain.images
            self.gt_normals_dir = self.config.train.datasetsTrain.gt_normals
            self.depth_dir = self.config.train.datasetsTrain.depth
            self.mask_dir = self.config.train.datasetsTrain.mask
            self.json_dir = self.config.train.datasetsTrain.json
        elif self.mode == 'test':
            self.images_dir = self.config.train.datasetsTestSynthetic.images
            self.gt_normals_dir = self.config.train.datasetsTestSynthetic.gt_normals
            self.depth_dir = self.config.train.datasetsTestSynthetic.depth
            self.mask_dir = self.config.train.datasetsTestSynthetic.mask
            self.json_dir = self.config.train.datasetsTestSynthetic.json
        else:
            raise KeyError('There is no such model!')
        self.axis = get_axis(self.config.train.axis)

    def __getitem__(self, index):
        # load datas
        _img, _gt_normals, _depth, _mask, _labels = self._load_datas(index)
        path = [self._datalist_input[index]]
        datas = copy.copy(self._data)
        datas['paths'] = path
        if 'real' in path:
            source = 'real'
        else:
            source = 'synthetic'
        datas['source'] = source
        if _img.size == (self.width_px, self.height_px):
            xmap = self.xmap
            ymap = self.ymap
            fx, fy, cx, cy = self.get_intrinsic(source, _img.size)
        elif _img.size == (self.width_px_2, self.height_px_2):
            xmap = self.xmap_2
            ymap = self.ymap_2
            fx, fy, cx, cy = self.get_intrinsic(source, _img.size)
        else:
            print('error data of size', path)
            datas = self._error_data
            datas['paths'] = path
            return datas

        width, height = _img.size[0], _img.size[1]
        self.ori_img = np.asarray(_img)
        instance_count = _labels['variants']['instance_count']
        camera_matrix = getCameraMatrix(_labels)

        ins_num = []
        for j in range(100, 100 + instance_count):
            ins_num.append(np.sum(_mask == j))

        # 面积太小就去掉
        ins_num = np.array(ins_num)
        vis_count = instance_count - sum(ins_num == 0)
        avg = sum(ins_num) / vis_count
        res = np.where(ins_num < avg * 0.4)
        res = np.array(res)
        if self.add_noise:
            _img = self.trancolor(_img)

        _img = np.array(_img)
        _gt_normals_tensor = torch.from_numpy(_gt_normals).permute((2, 0, 1))
        # 去掉背景
        # mask_back = ma.getmaskarray(ma.masked_not_equal(_mask, 0))
        # _gt_normals_tensor = _gt_normals_tensor.permute((2, 0, 1)) * mask_back

        objid = get_objid(self._datalist_input[index])
        model_points = self.pts['{}'.format(objid)]

        # plt.figure()
        # plt.imshow(self.ori_img)
        # plt.show()
        for i in range(100, 100 + instance_count):
            var_index = i - 100
            # 面积被遮住太多的不要
            if var_index in res:
                # print('The Object is Too Small, Pass!', i)
                continue

            variant_matrix = _labels['variants']['masks_and_poses_by_pixel_value'][str(i)]['world_pose']['matrix_4x4']

            mask_label = np.ma.getmaskarray(np.ma.masked_equal(_mask, np.array(i)))

            rmin, rmax, cmin, cmax = get_bbox(mask_label, width, height)

            # print('**********', max(rmax-rmin, cmax-cmin))
            if max(rmax-rmin, cmax-cmin) > self.max_img:
                print('The Object is Too Large, Pass!', i)
                continue

            # 太大的不要
            # if (rmax-rmin)*(cmax-cmin) > 150000:
            #     continue
            mask_croped = mask_label[rmin:rmax, cmin:cmax].astype(np.uint8)
            contours, _ = cv2.findContours(mask_croped*255, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
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

            datas['xmaps'].append(torch.from_numpy(xmap[rmin:rmax, cmin:cmax].astype(np.float32)).unsqueeze(dim=0))
            datas['ymaps'].append(torch.from_numpy(ymap[rmin:rmax, cmin:cmax].astype(np.float32)).unsqueeze(dim=0))

            img_masked = _img[rmin:rmax, cmin:cmax, :]
            # img_masked = torch.from_numpy(_img[rmin:rmax, cmin:cmax, :].transpose((2, 0, 1)).astype(np.float32))
            datas['img_cropeds'].append(self.norm(img_masked))

            dellist = [j for j in range(0, len(model_points))]
            dellist = random.sample(dellist, len(model_points) - self.num_pt)
            model_points = np.delete(model_points, dellist, axis=0)
            datas['model_points'].append(model_points)
            trans = np.dot(np.linalg.inv(camera_matrix), variant_matrix)

            if objid == 3:
                trans[0:3, 0:3] = trans[0:3, 0:3] * 10.0
            trans[1:3] = -trans[1:3]
            target_r = trans[0:3, 0:3]
            target_t = trans[0:3, 3]

            target = np.dot(model_points, target_r.T) + target_t
            # 画图的时候用
            # u = target[:, 0] * fx / target[:, 2] + cx
            # v = target[:, 1] * fy / target[:, 2] + cy
            # plt.scatter(u, v, s=0.1, alpha=0.5)

            datas['targets'].append(torch.from_numpy(target.astype(np.float32)))
            datas['obj_ids'].append(torch.LongTensor([int(objid)]))

            depth_masked = _depth[rmin:rmax, cmin:cmax]

            cam_scale = 1.0
            # d_scale = cam_scale * max([rmax-rmin, cmax-cmin]) / self.diameters[objid]
            d_scale = cam_scale * (rmax-rmin) * (cmax-cmin) / (width*height)

            depth_nromalized = depth_masked / d_scale

            datas['d_scales'].append(torch.FloatTensor([d_scale]))
            datas['depths'].append(torch.from_numpy(depth_nromalized.astype(np.float32)).unsqueeze(dim=0))

            normal_masked = _gt_normals_tensor[:, rmin:rmax, cmin:cmax]
            normal_masked = torch.nn.functional.normalize(normal_masked.double(), p=2, dim=0)
            datas['normals'].append(normal_masked)

            datas['intrinsic'].append(torch.FloatTensor([fx, fy, cx, cy]))
            datas['bboxes'].append(torch.FloatTensor([rmin, rmax, cmin, cmax]))
            datas['axis'].append(torch.from_numpy(self.axis[f'{objid}'].astype(np.float32)))
            datas['target_rs'].append(torch.from_numpy(target_r.astype(np.float32)))
            datas['target_ts'].append(torch.from_numpy(target_t.astype(np.float32)))

        del _img, mask_label, _depth, _mask, _gt_normals_tensor
        if not datas['model_points']:
            print('error data of none', path)
            datas = self._error_data
            datas['paths'] = path

        return datas

    def __len__(self):
        return len(self._datalist_depth)

    def _load_datas(self, index):
        """load RGB Mask Depth and Labels"""
        _img = Image.open(self._datalist_input[index]).convert('RGB')
        # _gt_normals: [H, W, C], C: RGB Channel Order
        _gt_normals = cv2.imread(self._datalist_gt_normals[index], cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()

        # _depth _mask [H, W]
        _depth = cv2.imread(self._datalist_depth[index], cv2.IMREAD_UNCHANGED)[:, :, 0]
        _mask = cv2.imread(self._datalist_mask[index], cv2.IMREAD_UNCHANGED)[:, :, 0]
        with open(self._datalist_json[index], encoding='utf-8') as _json:
            _labels = json.loads(_json.read())

        return _img, _gt_normals, _depth, _mask, _labels

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

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
        return {
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

    @property
    def res_name(self):
        return {
            '0': 'cup',
            '1': 'flower',
            '2': 'heart',
            '3': 'square',
            '4': 'stemless',
        }

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

    def get_intrinsic(self, source, imgsiz):
        if source == 'synthetic' and imgsiz == (self.width_px, self.height_px):
            fx, fy, cx, cy = 1386.42, 1386.46, 960.0, 540.0
        elif source == 'synthetic' and imgsiz == (self.width_px_2, self.height_px_2):
            fx, fy, cx, cy = 739.42, 739.44, 512.0, 288.0
        else:
            raise KeyError('Unknow source or imgsiz!')
        return fx, fy, cx, cy


class BathPoseDataset(PoseDataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine, config=None):
        super(BathPoseDataset, self).__init__(mode, num_pt, add_noise, root, noise_trans, refine, config)
        self.img_size = 256
        self.cam_scale = 1.0
        self.x_map = np.array([[j for i in range(self.img_size)] for j in range(self.img_size)])
        self.y_map = np.array([[i for i in range(self.img_size)] for j in range(self.img_size)])

    def __getitem__(self, index):
        # load datas
        _img, _gt_normals, _depth, _mask, _labels = self._load_datas(index)
        path = [self._datalist_input[index]]
        datas = dict()
        datas['flag'] = True
        if 'real' in path:
            source = 'real'
        else:
            source = 'synthetic'
        datas['source'] = source
        fx, fy, cx, cy = self.get_intrinsic(source, _img.size)
        width, height = _img.size[0], _img.size[1]
        self.ori_img = np.asarray(_img)

        instance_count = _labels['variants']['instance_count']
        camera_matrix = getCameraMatrix(_labels)

        ins_num = []
        for j in range(100, 100 + instance_count):
            ins_num.append(np.sum(_mask == j))
        # 面积太小就去掉
        ins_num = np.array(ins_num)
        vis_count = instance_count - sum(ins_num == 0)
        avg = sum(ins_num) / vis_count
        res = np.where(ins_num < avg * 0.4)
        res = np.array(res)

        # 添加噪声
        if self.add_noise:
            _img = self.trancolor(_img)

        _img = np.array(_img)
        _gt_normals_tensor = torch.from_numpy(_gt_normals)
        # 去掉背景
        # mask_back = ma.getmaskarray(ma.masked_not_equal(_mask, 0))
        # _gt_normals_tensor = _gt_normals_tensor.permute((2, 0, 1)) * mask_back

        objid = get_objid(self._datalist_input[index])
        model_points = self.pts['{}'.format(objid)]

        # 随机选择一个物体
        while True:
            random_obj = np.random.randint(100, 100 + instance_count)
            var_index = random_obj - 100
            if var_index in res:
                # print('The Object is Too Small, Pass!', i)
                continue
            else:
                break

        variant_matrix = _labels['variants']['masks_and_poses_by_pixel_value'][str(random_obj)]['world_pose']['matrix_4x4']
        trans = np.dot(np.linalg.inv(camera_matrix), variant_matrix)

        if objid == 3:
            trans[0:3, 0:3] = trans[0:3, 0:3] * 10.0
        trans[1:3] = -trans[1:3]
        target_r = trans[0:3, 0:3]
        target_t = trans[0:3, 3]

        mask_label = np.ma.getmaskarray(np.ma.masked_equal(_mask, np.array(random_obj)))
        rmin, rmax, cmin, cmax = get_square_bbox(mask_label, width, height)

        mask_croped = mask_label[rmin:rmax, cmin:cmax].astype(np.uint8)
        img_masked = _img[rmin:rmax, cmin:cmax, :]
        depth_masked = _depth[rmin:rmax, cmin:cmax]
        normal_masked = _gt_normals_tensor[rmin:rmax, cmin:cmax, :]
        normal_masked = torch.nn.functional.normalize(normal_masked.double(), p=2, dim=-1)
        ori_normal_masked = normal_masked.clone()
        # datas['ori_normal'] = ori_normal_masked
        del _img, mask_label, _depth, _mask, _gt_normals_tensor

        img_masked = cv2.resize(img_masked, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask_croped = cv2.resize(mask_croped, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        normal_masked = cv2.resize(normal_masked.numpy(), (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        depth_masked = cv2.resize(depth_masked, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        s_zoom = self.img_size / (rmax - rmin)
        # d_scale = s_zoom * self.cam_scale * (rmax - rmin) * (cmax - cmin) / (width * height)
        d_scale = self.img_size * self.cam_scale / (rmax - rmin)

        depth_normalized = depth_masked / d_scale
        xmap = self.x_map + rmin * s_zoom
        ymap = self.y_map + cmin * s_zoom

        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt)
        model_points = np.delete(model_points, dellist, axis=0)
        target = np.dot(model_points, target_r.T) + target_t

        # fixme: 不用的时候注释
        # datas['ori_size'] = torch.from_numpy(np.asarray(list(self.ori_img.shape)))
        # self.ori_img = cv2.resize(self.ori_img, (self.width_px, self.height_px))
        # datas['ori_img'] = torch.from_numpy(self.ori_img)
        datas['path'] = path
        # datas['img_cropeds'] = img_masked

        datas['img_cropeds'] = self.norm(img_masked)
        datas['masks'] = torch.from_numpy(mask_croped.astype(np.float32)).unsqueeze(dim=0)

        datas['normals'] = torch.from_numpy(normal_masked.astype(np.float32)).permute(2, 0, 1)
        datas['depths'] = torch.from_numpy(depth_normalized.astype(np.float32)).unsqueeze(dim=0)
        datas['d_scales'] = torch.FloatTensor([d_scale])
        datas['s_zoom'] = torch.FloatTensor([s_zoom])
        datas['xmaps'] = torch.from_numpy(xmap.astype(np.float32)).unsqueeze(dim=0)
        datas['ymaps'] = torch.from_numpy(ymap.astype(np.float32)).unsqueeze(dim=0)
        datas['model_points'] = model_points
        datas['target_rs'] = torch.from_numpy(target_r.astype(np.float32))
        datas['target_ts'] = torch.from_numpy(target_t.astype(np.float32))
        datas['targets'] = torch.from_numpy(target.astype(np.float32))
        datas['intrinsic'] = torch.FloatTensor([s_zoom*fx, s_zoom*fy, s_zoom*cx, s_zoom*cy])
        datas['obj_ids'] = torch.LongTensor([int(objid)])
        datas['axis'] = torch.from_numpy(self.axis[f'{objid}'].astype(np.float32))

        return datas

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


def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)
    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)
    return dst_img


def get_axis(src):
    axises = {}
    models_name = [
        'cup-with-waves',
        'flower-bath-bomb',
        'heart-bath-bomb',
        'square-plastic-bottle',
        'stemless-plastic-champagne-glass'
    ]

    for i, mn in enumerate(models_name):
        axises[f'{i}'] = axis(src[mn])
    return axises


def axis(src):
    ax = np.array([0., 0., 0.])
    if 'X' in src:
        ax[0] = 1
    if 'Y' in src:
        ax[1] = 1.
    if 'Z' in src:
        ax[2] = 1.
    return ax


def get_model(path, num_pt):
    """
    load points from .obj file
    :param path: dataset root path
    :return: pt:{
        '1': np.array([10000, 3])
        ....
        '5': np.array([10000, 3])
    }
    """
    pt = {}
    models_name = ['cup-with-waves.obj', 'flower-bath-bomb.obj', 'heart-bath-bomb.obj', 'square-plastic-bottle.obj', 'stemless-plastic-champagne-glass.obj']
    for i, mn in enumerate(models_name):
        mn = f'{path}/models/{mn}'
        points = sample_points_from_mesh(mn, num_pt)
        # print(f'{i}:', get_diameter(points))
        pt[f'{i}'] = torch.from_numpy(np.asarray(points.astype(np.float32)))
    return pt


def get_diameter(points):
    """ The two furthest points are the diameters。
    """
    distances = pairwise_distance(points, points)
    indx = np.argmax(distances)
    r = indx // distances.shape[1]
    c = indx % distances.shape[1]
    fps = points[[r, c]]
    return np.sqrt(np.sum((fps[0]-fps[1])**2))


def getCameraMatrix(dict):
    camera_matrix = np.array(dict['camera']['world_pose']['matrix_4x4'])
    camera_quaternion = dict['camera']['world_pose']['rotation']['quaternion']
    camera_matrix[:3, :3] = quaternion_to_rotation_matrix(np.array(camera_quaternion)).T[:3, :3]
    return camera_matrix


def quaternion_to_rotation_matrix(quat):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], 0.0],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], 0.0],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=q.dtype)
    return rot_matrix


# border_list = [-1, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 440, 480, 520,
#                560, 600, 640, 680, 800, 1000]
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680,
               720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320,
               1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920]


def get_bbox(label, img_width, img_length):
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


def get_square_bbox(label, width_px, height_px):
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


def get_objid(path):
    if 'cup' in path:
        return 0
    if 'flower' in path:
        return 1
    if 'heart' in path:
        return 2
    if 'square' in path:
        return 3
    if 'stemless' in path:
        return 4
    return -1


if __name__ == '__main__':
    dataset = PoseDataset('test', 1000, False, '/root/Source/ymy_dataset/cleargrasp', 0.03, False)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)
    data_iters = dataloader.__iter__()
    while True:
        ds = data_iters.__next__()

    # print(ds['img_cropeds'].shape)
    # i = np.random.randint(0, 8)
    # imgs = ds['img_cropeds'][i]
    # # img_ori = ds['ori_img'][i]
    # s_zoom = ds['s_zoom'][i]
    # target = ds['targets'][i]
    # intrinsic = ds['intrinsic'][i]
    # print(
    #     ds['img_cropeds'].shape,
    #     ds['intrinsic'].shape,
    #     ds['xmaps'].shape,
    #     ds['ymaps'].shape,
    #     ds['d_scales'].shape,
    #     ds['obj_ids'].shape,
    # )

    # fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    # # fx, fy, cx, cy = intrinsic[0]/s_zoom, intrinsic[1]/s_zoom, intrinsic[2]/s_zoom, intrinsic[3]/s_zoom
    # ori_size = ds['ori_size'][i]
    # print('ori', ori_size)
    # ori_img = None
    #
    # if ori_size[0] == 1080:
    #     h, w, _ = img_ori.shape
    #     ori_img = img_ori.clone()
    #     img_ori = cv2.resize(img_ori.numpy(), (int(w*s_zoom), int(h*s_zoom)))
    # elif ori_size[0] == 576:
    #     img_ori = cv2.resize(img_ori.numpy(), (1024, 576))
    #     h, w, _ = img_ori.shape
    #     ori_img = img_ori.copy()
    #     img_ori = cv2.resize(img_ori, (int(w*s_zoom), int(h*s_zoom)))
    #
    # print('resize', img_ori.shape)
    # plt.figure()
    # plt.axis('off')
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.subplot(121)
    # plt.imshow(ori_img)
    # u = target[:, 0] * fx / target[:, 2] + cx
    # v = target[:, 1] * fy / target[:, 2] + cy
    # plt.scatter(u, v, s=0.05, alpha=0.5)
    #
    # plt.subplot(122)
    # plt.axis('off')
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.imshow(img_ori)
    # plt.scatter(u, v, s=0.05, alpha=0.5)
    # plt.show()

    # for i in range(imgs.size(0)):
    #     plt.subplot(2, 5, i+1)
    #     plt.axis('off')
    #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #     plt.imshow(imgs[i].numpy())
    # plt.show()
    # while True:
    #     ds = data_iters.__next__()
    #     for idx in range(len(ds['img_cropeds'])):
    #         bbox = ds['bboxes'][idx]
    #         print(
    #             (bbox[0, 1]-bbox[0, 0])*(bbox[0, 3]-bbox[0, 2])
    #         )
    #
    #     while True:
    #         if cv2.waitKey():
    #             break
        #
        # if not ds['flag'].item():
        #     print('error data')
        #     continue
        # else:
        #     print('right data')
        #     continue

    # datas = data_iters.__next__()
    # ori_img = dataset.ori_img
    # # img_croped = datas['img_cropeds'][0].squeeze().permute((1, 2, 0))
    # bbox = datas['bboxes'][0].squeeze().int()
    # img_croped = ori_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # mask = datas['masks'][0].squeeze()
    # depth = datas['depths'][0].squeeze()
    # normal = datas['normals'][0].squeeze().permute((1, 2, 0))
    # plt.figure()
    #
    # plt.subplot(221)
    # plt.axis('off')
    # plt.imshow(img_croped)
    # plt.subplot(222)
    # plt.axis('off')
    # plt.imshow(mask)
    # plt.subplot(223)
    # plt.axis('off')
    # plt.imshow(depth)
    # plt.subplot(224)
    # plt.axis('off')
    # plt.imshow(normal)
    # plt.show()
    # datas = {
    #     'img_cropeds': [], # [[bs, 3, H, W], ]
    #     'intrinsic': [],
    #     'depths': [],
    #     'bboxes': [],
    #     'obj_ids': [],
    #     'targets': [],
    #     'd_scales': [],
    #     'model_points': [],
    #     'masks': [],
    #     'normals': [],
    #     'flag': True
    # }
