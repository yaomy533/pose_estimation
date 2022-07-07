#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/19 16:54
# @Author  : yaomy

# real+render+fuse 所有物体
import time
import copy
import torch
import random
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import pickle as pkl
import numpy as np
import yaml
import os
import pickle
from pathlib import Path
from PIL import Image
import cv2
from mmcv import Config
from tqdm import tqdm
import numpy.ma as ma
from lib.transform.coordinate import crop_resize_by_warp_affine


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETCT_PATH = Path(os.path.realpath(__file__)).parent.parent.parent
CONFIG = Config.fromfile(f'{PROJETCT_PATH}/config/linemod/lm_v3_1.py')


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, num_point, add_noise, root, noise_trans, num_kps, cls_type=None, cfg=CONFIG):
        self.obj_dict = {
            'ape': 1, 'benchvise': 2, 'bowl': 3, 'cam': 4, 'can': 5, 'cat': 6, 'cup': 7, 'driller': 8,
            'duck': 9, 'eggbox': 10, 'glue': 11, 'holepuncher': 12, 'iron': 13, 'lamp': 14, 'phone': 15,
        }

        self.obj_name = {v: k for k, v in self.obj_dict.items()}
        if cls_type is None or cls_type == 'all':
            self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        else:
            self.objlist = [self.obj_dict[cls_type]]

        print(f'{mode} Use Object: ', self.objlist)
        self.mode = mode
        self.num_point = num_point
        self.num_kps = num_kps
        self.cfg = cfg

        self.num_pt_mesh = num_point
        self.num_pt_mesh_large = 2600
        self.num_pt_sample = 5000
        self.num_pt_mesh_small = num_point

        self.noise_trans = noise_trans
        self.add_noise = add_noise
        self.height = 480.0
        self.width = 640.0
        self.scale = 1000.0
        self.num_syn = self.cfg.Data.NUM_SYN
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])

        self.data_sample = [i for i in range(9600, 8)]
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)

        # 要归一化后才能用这个
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sym_obj = [7, 8]
        self.obj_dict_name = {v: k for k, v in self.obj_dict.items()}

        self.intrinsic_matrix = {
            'linemod': np.array([[572.4114, 0., 325.2611],
                                 [0., 573.57043, 242.04899],
                                 [0., 0., 1.]]),

            'blender': np.array([[700., 0., 320.],
                                 [0., 700., 240.],
                                 [0., 0., 1.]]),
        }

        self.ori_iamge = np.zeros((480, 640))
        self.gt_pose = torch.zeros((4, 4))
        self.root = root
        self.synthetic_all = []
        self.real_all = []
        self.diameter = []
        self.pt = {}
        self.extent = {}
        self.lf_border = {}
        self.meta = {}
        self.kps = {}
        self.centers = {}
        with open(f'{CURRENT_DIR}/dataset_config/models_info.yml', 'r') as info_yaml:
            info = yaml.load(info_yaml, Loader=yaml.FullLoader)
        self.info = info

        with open(f'{CURRENT_DIR}/dataset_config/fps_64.pkl', 'rb') as file_fps:
            fps = pickle.load(file_fps)

        self.fps = fps
        pbar = tqdm(self.objlist)
        for item in pbar:
            st = time.time()
            cls_root = os.path.join(self.root, 'data', f'{item:02d}')
            if self.mode == 'train':
                cls_type = self.obj_dict_name[item]
                real_img_pth = os.path.join(cls_root, "train.txt")
                rnd_img_pth = os.path.join(self.root, f"renders/{cls_type}/file_list.txt")
                fuse_img_pth = os.path.join(self.root, f"fuse/{cls_type}/file_list.txt")

                rnd_img_part = os.path.join(self.root, f"renders/{cls_type}/file_list_part_5000.txt")
                fuse_img_part = os.path.join(self.root, f"fuse/{cls_type}/file_list_part_5000.txt")

                # render path
                rnd_lst = read_lines(rnd_img_pth)
                # fuse path
                fuse_lst = read_lines(fuse_img_pth)
                # real path
                real_lst = read_lines(real_img_pth)

                # render path part
                rnd_lst_part = random.sample(read_lines(rnd_img_part), self.cfg.Data.NUM_SYN)
                # fuse path part
                fuse_lst_part = random.sample(read_lines(fuse_img_part), self.cfg.Data.NUM_SYN)

                real_lst_bat = real_lst[:]

                if self.cfg.Data.PART_SYN:
                    for i in range(2):
                        real_lst += real_lst_bat
                    synthetic = rnd_lst_part + fuse_lst_part
                else:
                    for i in range(10):
                        real_lst += real_lst_bat
                    synthetic = fuse_lst + rnd_lst

                self.synthetic_all += synthetic
                self.real_all += [{'cls_id': item, 'im_id': real_id} for real_id in real_lst]
            else:
                tst_img_pth = os.path.join(cls_root, "test.txt")
                real_lst = read_lines(tst_img_pth)
                self.real_all += [{'cls_id': item, 'im_id': real_id} for real_id in real_lst]

            with open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r') as meta_file:
                self.meta[item] = yaml.load(meta_file, Loader=yaml.FullLoader)

            # self.pt[item], _, _ = self._load_model('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))

            # fixme: pkl是均匀采样后的点
            with open(f'{self.root}/models/obj_{item:02d}.pkl', 'rb') as f:
                mdl = pickle.load(f) / 1000.0
                
            self.pt[item] = mdl

            self.extent[item] = np.asarray([info[item]['size_x'], info[item]['size_y'], info[item]['size_z']]) / 1000.0
            self.lf_border[item] = np.asarray([info[item]['min_x'], info[item]['min_y'], info[item]['min_z']]) / 1000.0
            self.diameter.append(info[item]['diameter'] / 1000.0)
            pbar.set_description(f'item : {item}, cost: {time.time() - st: .02f}')

        self.all_lst = self.real_all + self.synthetic_all
        # self.all_lst = self.synthetic_all

        print(f'len of real {self.mode}: ', len(self.real_all))
        print(f'len of synthetic {self.mode}: ', len(self.synthetic_all))
        print(f'len of all {self.mode}: ', len(self.all_lst), '\n')

    def random_syn(self):
        synthetic_all = []
        for item in self.objlist:
            cls_type = self.obj_dict_name[item]

            rnd_img_pth = os.path.join(self.root, f"renders/{cls_type}/file_list.txt")
            fuse_img_pth = os.path.join(self.root, f"fuse/{cls_type}/file_list.txt")

            # render path
            rnd_lst = read_lines(rnd_img_pth)
            # fuse path
            fuse_lst = read_lines(fuse_img_pth)

            synthetic = random.sample(fuse_lst, self.num_syn) + random.sample(rnd_lst, self.num_syn)
            synthetic_all += synthetic

        self.all_lst = self.real_all + synthetic_all

    def _load_real_data(self, item):
        im_id = item['im_id']
        cls_id = item['cls_id']
        cls_root = os.path.join(self.root, 'data', f'{cls_id:02d}')
        with Image.open(os.path.join(cls_root, "depth/{}.png".format('%04d' % int(im_id)))) as di:
            depth = np.asarray(di)

        with open(f'{cls_root}/normal/{int(im_id):04d}-normal.pkl', 'rb') as f_n:
            normal = pickle.load(f_n)

        # with open(f'{cls_root}/normal/{int(im_id):04d}-normal-croped.pkl', 'rb') as f_n:
        #     croped_normal = pickle.load(f_n)

        with open(f'{cls_root}/xyz/{int(im_id):04d}-coordinate.pkl', 'rb') as f_c:
            coordinate = pickle.load(f_c)

        with open(f'{cls_root}/xyz/{int(im_id):04d}-region.pkl', 'rb') as f_r:
            region = pickle.load(f_r)

        if self.mode == 'eval':
            with Image.open(
                    os.path.join(self.root, "segnet_results", f'{int(cls_id):02d}_label/{int(im_id):04d}_label.png')
            ) as li:
                label = np.array(li)
        else:
            with Image.open(os.path.join(cls_root, "mask/{}.png".format('%04d' % int(im_id)))) as li:
                label = np.array(li)

        with Image.open(os.path.join(cls_root, "rgb/{}.png".format('%04d' % int(im_id)))) as ri:
            # if self.add_noise:
            #     ri = self.trancolor(ri)
            img = np.array(ri)[:, :, :3]

        self.ori_image = img.copy()
        cam_scale = 1000.0
        meta = self.meta[int(cls_id)][int(im_id)]
        if cls_id == 2:
            for i in range(0, len(meta)):
                if meta[i]['obj_id'] == 2:
                    meta = meta[i]
                    break
        else:
            meta = meta[0]

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c']) / cam_scale
        self.gt_pose = self.rt2matrix(target_r, target_t)
        bbox = meta['obj_bb']
        if self.mode != 'eval':
            mask_label = np.ma.getmaskarray(np.ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        else:
            mask_label = np.ma.getmaskarray(np.ma.masked_equal(label, np.array(255)))
        # mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))
        # mask_label = mask_label
        return {
            'img': img,
            'depth': depth / cam_scale,
            'normal': normal,
            'coordinate': coordinate,
            'region': region,
            # 'croped_normal': croped_normal,
            'mask_label': mask_label,
            'cls_id': cls_id,
            'K': self.intrinsic_matrix["linemod"],
            'bbox': bbox,
            'cam_scale': cam_scale,
            'target_r': target_r,
            'target_t': target_t,
            'type': 'real'
        }

    def _load_syn_data(self, item):
        pkl_path = os.path.join(self.root, item)
        if 'renders' in item:
            tp = 'renders'
        elif 'fuse' in item:
            tp = 'fuse'
        else:
            raise KeyError
        with open(pkl_path, "rb") as f:
            data = pkl.load(f)
        depth = data['depth']
        img = data['rgb']

        self.ori_image = img.copy()
        # if self.add_noise:
        #     PIL_image = Image.fromarray(img)
        #     img = self.trancolor(PIL_image)
        img = np.array(img)[:, :, :3]

        labels = data['mask']
        # K = data['K']
        RT = data['RT']
        self.gt_pose = RT
        target_r = RT[:, :3]
        target_t = RT[:, 3]
        cls_type = item.split('/')[-2]
        cls_id = copy.copy(self.obj_dict[cls_type])
        rnd_typ = data['rnd_typ']

        if rnd_typ == "fuse":
            label = (labels == cls_id).astype("uint8")
        else:
            label = (labels > 0).astype("uint8")

        # with open(os.path.splitext(pkl_path)[0] + '-normal-croped.pkl', 'rb') as f_n:
        #     croped_normal = pickle.load(f_n)

        with open(f'{self.root}/{item.split(".")[0]}-normal.pkl', 'rb') as f_n:
            normal = pickle.load(f_n)

        with open(f'{self.root}/{item.split(".")[0]}-coordinate.pkl', 'rb') as f_n:
            coordinate = pickle.load(f_n)

        with open(f'{self.root}/{item.split(".")[0]}-region.pkl', 'rb') as f_n:
            region = pickle.load(f_n)

        cam_scale = 1.0
        mask_label = np.ma.getmaskarray(np.ma.masked_equal(label, np.array(1)))
        index = np.where(mask_label == True)
        try:
            rmax = index[0].max()
            rmin = index[0].min()
            cmax = index[1].max()
            cmin = index[1].min()
            bbox = [cmin, rmin, cmax - cmin, rmax - rmin]
        except ValueError:
            return None

        return {
            'img': img,
            'depth': depth,
            'normal': normal,
            'coordinate': coordinate,
            # 'croped_normal': croped_normal,
            'region': region,
            'mask_label': mask_label,
            'cls_id': cls_id,
            'K': data['K'],
            'bbox': bbox,
            'cam_scale': cam_scale,
            'target_r': target_r,
            'target_t': target_t,
            'type': tp
        }

    def _load_resize_data(self, item):
        if type(item).__name__ == 'dict':
            while True:
                ds = self._load_real_data(item)
                if ds is None:
                    print(f'Real data error pass ! {item}')
                    item = random.choice(self.real_all)
                else:
                    break
        else:
            while True:
                ds = self._load_syn_data(item)
                if ds is None:
                    print(f'Syn data error pass ! {item}')
                    item = random.choice(self.synthetic_all)
                else:
                    break

        out_size = self.cfg.Data.OUT_SIZE
        input_size = self.cfg.Data.INPUT_SIZE
        resize_type = self.cfg.Data.RESIZE_TYPE
        rmin, rmax, cmin, cmax = get_square_bbox(ds['bbox'])

        img_masked = np.array(ds['img'])[rmin:rmax, cmin:cmax, :]
        cam_fx, cam_fy, cam_cx, cam_cy = ds['K'][0, 0], ds['K'][1, 1], ds['K'][0, 2], ds['K'][1, 2]

        cls_id = ds['cls_id']
        target_r, target_t = ds['target_r'], ds['target_t']

        if ds['normal'].shape == (self.height, self.width, 3):
            normal_ = ds['normal'][rmin:rmax, cmin:cmax, :]
        else:
            normal_ = ds['normal']

        hole_normal = np.ones((int(self.height), int(self.width), 3))
        hole_normal[rmin:rmax, cmin:cmax, :] = normal_
        normal_masked = hole_normal * (hole_normal != [1, 1, 1])

        if resize_type == 'resize':
            resize_normal = cv2.resize(normal_masked, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        else:
            resize_normal= crop_resize_by_warp_affine(
                normal_masked, [(cmin + cmax) / 2, (rmin + rmax) / 2], int(rmax - rmin), out_size, interpolation=cv2.INTER_NEAREST
            )

        # croped_normal_ = ds['croped_normal']
        # croped_normal = croped_normal_ * (croped_normal_ != [1, 1, 1])
        #
        # if resize_type == 'resize':
        #     resize_croped_normal = cv2.resize(croped_normal, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        # else:
        #     resize_croped_normal= crop_resize_by_warp_affine(
        #         croped_normal, [(cmin + cmax) / 2, (rmin + rmax) / 2], int(rmax - rmin), 64, interpolation=cv2.INTER_NEAREST
        #     )

        if ds['coordinate'].shape == (self.height, self.width, 3):
            coordinate = ds['coordinate'][rmin:rmax, cmin:cmax, :]
        else:
            coordinate = ds['coordinate']

        # 变成[0, 1]
        # coordinate_map = np.ones_like(coordinate)
        hole_coordinate = np.zeros((int(self.height), int(self.width), 3))
        hole_coordinate[rmin:rmax, cmin:cmax, :] = coordinate
        coordinate_map = np.zeros_like(hole_coordinate)
        lf_border = self.lf_border[cls_id]
        extent = self.extent[cls_id]

        mask_obj = (hole_coordinate != [0, 0, 0])[..., 0]
        mask = (ds['mask_label']*mask_obj).astype(np.float32)

        if resize_type == 'resize':
            resize_mask = cv2.resize(mask[:, :, None], (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        else:
            resize_mask = crop_resize_by_warp_affine(
                mask[:, :, None], [(cmin + cmax) / 2, (rmin + rmax) / 2], int(rmax - rmin), out_size,
                interpolation=cv2.INTER_NEAREST
            )

        choose = mask[rmin:rmax, cmin:cmax, None].flatten().nonzero()[0]

        if len(choose) > self.num_point:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_point] = 1
            np.random.shuffle(c_mask)
            # num_points+8
            choose = choose[c_mask.nonzero()]
        else:
            try:
                choose = np.pad(choose, (0, self.num_point - len(choose)), 'wrap')
            except ValueError:
                print('choose error:', item)
                return None
        choose = np.array([choose])

        resize_choose = resize_mask.flatten().nonzero()[0]
        if len(resize_choose) > self.num_point:
            c_mask = np.zeros(len(resize_choose), dtype=int)
            c_mask[:self.num_point] = 1
            np.random.shuffle(c_mask)
            # num_points+8
            resize_choose = resize_choose[c_mask.nonzero()]
        else:
            try:
                resize_choose = np.pad(resize_choose, (0, self.num_point - len(resize_choose)), 'wrap')
            except ValueError:
                print('choose error:', item)
                return None

        resize_choose = np.array([resize_choose])

        if resize_type == 'resize':
            hole_coordinate_map = cv2.resize(hole_coordinate, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        else:
            hole_coordinate_map = crop_resize_by_warp_affine(
                hole_coordinate, [(cmin + cmax) / 2, (rmin + rmax) / 2], int(rmax - rmin), out_size,
                interpolation=cv2.INTER_NEAREST
            )

        coordinate_choosed = hole_coordinate_map.reshape(-1, 3)[resize_choose[0]]
        coordinate_map[:, :, 0] = (hole_coordinate[:, :, 0] - lf_border[0]) / extent[0]
        coordinate_map[:, :, 1] = (hole_coordinate[:, :, 1] - lf_border[1]) / extent[1]
        coordinate_map[:, :, 2] = (hole_coordinate[:, :, 2] - lf_border[2]) / extent[2]
        coordinate_map = coordinate_map * (hole_coordinate != [0, 0, 0])

        if resize_type == 'resize':
            resize_coordinate_map = cv2.resize(coordinate_map, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        else:
            resize_coordinate_map = crop_resize_by_warp_affine(
                coordinate_map, [(cmin + cmax) / 2, (rmin + rmax) / 2], int(rmax - rmin), out_size,
                interpolation=cv2.INTER_NEAREST
            )

        if ds['region'].shape == (self.height, self.width, 3):
            region = ds['region'][rmin:rmax, cmin:cmax]
        else:
            region = ds['region']
        hole_region = np.zeros((int(self.height), int(self.width)))
        hole_region[rmin:rmax, cmin:cmax] = region
        if resize_type == 'resize':
            resize_region = cv2.resize(hole_region, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        else:
            resize_region = crop_resize_by_warp_affine(
                hole_region, [(cmin + cmax) / 2, (rmin + rmax) / 2], int(rmax - rmin), out_size,
                interpolation=cv2.INTER_NEAREST
            )

        ori_kps = self.kps[cls_id] / 1.0
        # ori_center = self.centers[cls_id] / 1.0
        ori_center = np.array([[0., 0., 0.]])
        all_kps = np.concatenate([ori_kps, ori_center], axis=0)

        kps = np.dot(ori_kps, target_r.T) + target_t
        center = np.dot(ori_center, target_r.T) + target_t
        kps = np.concatenate([kps, center], axis=0)
        u = kps[:, 0] * cam_fx / kps[:, 2] + cam_cx
        v = kps[:, 1] * cam_fy / kps[:, 2] + cam_cy

        # [8, 3]
        uvd1 = np.stack([u, v, kps[:, 2]], axis=1)

        model_points = self.pt[cls_id]

        target = model_points @ target_r.T + target_t
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for _ in range(3)])

        if self.add_noise:
            target = np.add(target, add_t)

        if resize_type == 'resize':
            reize_img = cv2.resize(img_masked/255., (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        else:
            reize_img = crop_resize_by_warp_affine(
                ds['img']/255., [(cmin+cmax)/2, (rmin+rmax)/2], int(rmax-rmin), input_size, interpolation=cv2.INTER_LINEAR
            )

        kps_scale = float(rmax-rmin) / self.cfg.Data.INPUT_SIZE
        resize_scale = float(rmax-rmin) / self.cfg.Data.OUT_SIZE

        resize_uvd = uvd1.copy()
        resize_uvd[..., 0] -= cmin
        resize_uvd[..., 1] -= rmin
        resize_uvd[..., :2] /= kps_scale

        resize_xmap = torch.LongTensor([[i for i in range(out_size)] for _ in range(out_size)]) / out_size
        resize_ymap = torch.LongTensor([[j for _ in range(out_size)] for j in range(out_size)]) / out_size

        resize_cam_fx, resize_cam_fy, resize_cam_cx, resize_cam_cy = cam_fx / resize_scale, cam_fy / resize_scale, out_size / 2, out_size / 2

        if resize_type == 'resize':
            x_map = cv2.resize(self.xmap[rmin:rmax, cmin:cmax], (out_size, out_size), cv2.INTER_LINEAR) / 640.
            y_map = cv2.resize(self.ymap[rmin:rmax, cmin:cmax], (out_size, out_size), cv2.INTER_LINEAR) / 480.
        else:
            x_map = crop_resize_by_warp_affine(
                self.xmap/640., [(cmin+cmax)/2, (rmin+rmax)/2], int(rmax-rmin), out_size, interpolation=cv2.INTER_NEAREST
            )

            y_map = crop_resize_by_warp_affine(
                self.ymap/480., [(cmin+cmax)/2, (rmin+rmax)/2], int(rmax-rmin), out_size, interpolation=cv2.INTER_NEAREST
            )

        depth = ds['depth']

        if resize_type == 'resize':
            resize_depth = cv2.resize(depth, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        else:
            resize_depth = crop_resize_by_warp_affine(
                depth, [(cmin + cmax) / 2, (rmin + rmax) / 2], int(rmax - rmin), out_size,
                interpolation=cv2.INTER_NEAREST
            )

        # depth_choosed = resize_depth.reshape(-1)[resize_choose[0]][:, np.newaxis].astype(np.float32)
        x_map_choosed = x_map.reshape(-1)[resize_choose[0]][:, np.newaxis].astype(np.float32) * 640.
        y_map_choosed = y_map.reshape(-1)[resize_choose[0]][:, np.newaxis].astype(np.float32) * 480.
        depth_choosed = depth[y_map_choosed.astype(np.int16), x_map_choosed.astype(np.int16)]

        pt2 = depth_choosed / 1.0
        pt0 = (x_map_choosed - cam_cx) * pt2 / cam_fx
        pt1 = (y_map_choosed - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        if self.add_noise:
            cloud = np.add(cloud, add_t)

        return {
            'target': torch.from_numpy(target.astype(np.float32)),  # [Np, 3]
            'cloud': torch.from_numpy(cloud.astype(np.float32)),
            'model_points': torch.from_numpy(model_points.astype(np.float32)),
            'cls_id': torch.LongTensor([self.objlist.index(cls_id)]),
            'choose': torch.LongTensor(choose.astype(np.int32)),
            'intrinsic': torch.FloatTensor([cam_fx, cam_fy, cam_cx, cam_cy]),
            'uvd1': torch.from_numpy(uvd1),  # [9, 3]
            'trans_kps': torch.from_numpy(kps.astype(np.float32)),
            'kps': torch.from_numpy(ori_kps.astype(np.float32)),
            'all_kps': torch.from_numpy(all_kps.astype(np.float32)),
            'trans_center': torch.from_numpy(center.astype(np.float32)),
            'center': torch.from_numpy(ori_center.astype(np.float32)),
            'target_r': torch.from_numpy(target_r.astype(np.float32)),
            'target_t': torch.from_numpy(target_t.astype(np.float32)),
            'bbox': torch.FloatTensor([rmin, rmax, cmin, cmax]),
            'type': ds['type'],
            'xmap': torch.from_numpy(x_map.astype(np.float32)).unsqueeze(dim=0),
            'ymap': torch.from_numpy(y_map.astype(np.float32)).unsqueeze(dim=0),
            'ori_mask': torch.from_numpy(ds['mask_label'].astype(np.float32)),
            'ori_depth': torch.from_numpy(ds['depth'].astype(np.float32)),
            'ori_coordinate': torch.from_numpy(hole_coordinate.astype(np.float32)),
            'coordinate_choosed': torch.from_numpy(coordinate_choosed.astype(np.float32)),
            'resize_img': torch.from_numpy(reize_img.astype(np.float32)).permute(2, 0, 1),
            'resize_normal': F.normalize(torch.from_numpy(resize_normal).permute(2, 0, 1), p=2, dim=0).float(),
            'resize_uvd': torch.from_numpy(resize_uvd),
            'resize_scale': torch.tensor(resize_scale),
            'kps_scale': torch.tensor(kps_scale),
            'resize_xmap': resize_xmap,
            'resize_ymap': resize_ymap,
            'resize_choose': torch.LongTensor(resize_choose.astype(np.int32)),
            'resize_intrinsic': torch.FloatTensor([resize_cam_fx, resize_cam_fy, resize_cam_cx, resize_cam_cy]),
            'resize_coordinate': torch.from_numpy((resize_coordinate_map*resize_mask[:, :, None]).astype(np.float32)).permute(2, 0, 1),
            'extent': torch.from_numpy(extent),  # [3]
            'lfborder': torch.from_numpy(lf_border),  # [3]
            # 'resize_croped_normal': F.normalize(torch.from_numpy(resize_croped_normal).permute(2, 0, 1), p=2, dim=0).float(),  # [3]
            'resize_mask': torch.from_numpy(resize_mask.astype(np.float32)).unsqueeze(dim=0),
            'resize_region': torch.from_numpy(resize_region*resize_mask).long(),
        }

    def _load_data(self, item):
        if type(item).__name__ == 'dict':
            while True:
                ds = self._load_real_data(item)
                if ds is None:
                    print(f'Real data error pass ! {item}')
                    item = random.choice(self.real_all)
                else:
                    break
        else:
            while True:
                ds = self._load_syn_data(item)
                if ds is None:
                    print(f'Syn data error pass ! {item}')
                    item = random.choice(self.synthetic_all)
                else:
                    break

        rmin, rmax, cmin, cmax = get_square_bbox(ds['bbox'])
        cam_fx, cam_fy, cam_cx, cam_cy = ds['K'][0, 0], ds['K'][1, 1], ds['K'][0, 2], ds['K'][1, 2]
        cls_id = ds['cls_id']
        depth = ds['depth'][rmin:rmax, cmin:cmax]
        img = np.array(ds['img'])
        target_r, target_t = ds['target_r'], ds['target_t']
        r_center = int((rmax + rmin) / 2)
        c_center = int((cmax + cmin) / 2)

        if ds['normal'].shape == (self.height, self.width, 3):
            normal_ = ds['normal'][rmin:rmax, cmin:cmax, :]
        else: # keep croped picture to save memory
            normal_ = ds['normal']
            h, w, _ = normal_.shape
            w_half, h_half = int(w/2), int(h/2)
            try:
                hole_normal = np.ones((int(self.height), int(self.width), 3))
                hole_normal[r_center-h_half:r_center+h_half, c_center-w_half:c_center+w_half, :] = normal_
                normal_ = hole_normal[rmin:rmax, cmin:cmax, :]
            except ValueError:
                return None

        if ds['coordinate'].shape == (self.height, self.width, 3):
            coordinate = ds['coordinate'][rmin:rmax, cmin:cmax, :]
        else:
            coordinate = ds['coordinate']
            h, w, _ = coordinate.shape
            w_half, h_half = int(w/2), int(h/2)
            hole_coordinate = np.ones((int(self.height), int(self.width), 3))
            hole_coordinate[r_center-h_half:r_center+h_half, c_center-w_half:c_center+w_half, :] = coordinate
            coordinate = hole_coordinate[rmin:rmax, cmin:cmax, :]

        if ds['region'].shape == (self.height, self.width, 3):
            region = ds['region'][rmin:rmax, cmin:cmax]
        else:
            region = ds['region']
            h, w = region.shape
            w_half, h_half = int(w/2), int(h/2)
            hole_region = np.ones((int(self.height), int(self.width)))
            hole_region[r_center-h_half:r_center+h_half, c_center-w_half:c_center+w_half] = region
            region = hole_region[rmin:rmax, cmin:cmax]

        normal_masked = normal_ * (normal_ != [1, 1, 1])
        lf_border = self.lf_border[cls_id].copy()
        extent = self.extent[cls_id].copy()

        mask_obj = (coordinate != [0, 0, 0])[..., 0]
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ds['mask_label'][rmin:rmax, cmin:cmax]
        mask = (mask_label * mask_obj * mask_depth).astype(np.float32)
        multi_cls_mask = mask * float(self.objlist.index(cls_id) + 1.)

        choose = mask.flatten().nonzero()[0]
        if len(choose) > self.num_point:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_point] = 1
            np.random.shuffle(c_mask)
            # num_points+8
            choose = choose[c_mask.nonzero()]
        else:
            try:
                choose = np.pad(choose, (0, self.num_point - len(choose)), 'wrap')
            except ValueError:
                print('choose error:', item)
                return None

        choose = np.array([choose])
        coordinate_choosed = coordinate.reshape(-1, 3)[choose[0]]
        xyz_map = np.zeros_like(coordinate)
        xyz_map[:, :, 0] = (coordinate[:, :, 0] - lf_border[0]) / extent[0]
        xyz_map[:, :, 1] = (coordinate[:, :, 1] - lf_border[1]) / extent[1]
        xyz_map[:, :, 2] = (coordinate[:, :, 2] - lf_border[2]) / extent[2]

        xyz_masked = xyz_map * (coordinate != [0, 0, 0])

        model_points = self.pt[cls_id]
        if self.mode == 'train':
            num_pt_mesh = self.num_pt_mesh
        else:
            num_pt_mesh = self.num_pt_mesh_large

        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - num_pt_mesh)
        model_points = np.delete(model_points, dellist, axis=0)

        model_points : np.array
        target = model_points @ target_r.T + target_t
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for _ in range(3)])

        if self.add_noise:
            target = np.add(target, add_t)
        x_map = self.xmap[rmin:rmax, cmin:cmax]
        y_map = self.ymap[rmin:rmax, cmin:cmax]
        depth_choosed = depth.reshape(-1)[choose[0]][:, np.newaxis].astype(np.float32)
        x_map_choosed = x_map.reshape(-1)[choose[0]][:, np.newaxis].astype(np.float32)
        y_map_choosed = y_map.reshape(-1)[choose[0]][:, np.newaxis].astype(np.float32)

        pt2 = depth_choosed / 1.0
        pt0 = (x_map_choosed - cam_cx) * pt2 / cam_fx
        pt1 = (y_map_choosed - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        img_croped = img[rmin:rmax, cmin:cmax] / 255.
        region_point = self.fps[cls_id].copy()

        region_point[:, 0] = (region_point[:, 0] - lf_border[0]) / extent[0]
        region_point[:, 1] = (region_point[:, 1] - lf_border[1]) / extent[1]
        region_point[:, 2] = (region_point[:, 2] - lf_border[2]) / extent[2]
        region_point = np.concatenate([np.asarray([[0, 0, 0]]), region_point])

        return {
            'model_points': torch.from_numpy(model_points.astype(np.float32)),
            'cls_id': torch.LongTensor([self.objlist.index(cls_id)]),
            'intrinsic': torch.FloatTensor([cam_fx, cam_fy, cam_cx, cam_cy]),
            'bbox': torch.FloatTensor([rmin, rmax, cmin, cmax]),
            'coordinate_choosed': torch.from_numpy(coordinate_choosed.astype(np.float32)),
            'type':ds['type'],
            # for PnP
            'xmap': torch.from_numpy(x_map.astype(np.float32)).unsqueeze(dim=0),
            'ymap': torch.from_numpy(y_map.astype(np.float32)).unsqueeze(dim=0),
            'x_map_choosed': torch.from_numpy(x_map_choosed.astype(np.float32)),
            'y_map_choosed': torch.from_numpy(y_map_choosed.astype(np.float32)),

            # input of model
            'img_croped': self.norm(torch.from_numpy(img_croped.astype(np.float32)).permute(2, 0, 1)),
            'choose': torch.LongTensor(choose.astype(np.int32)),
            'region_point': torch.from_numpy(region_point.astype(np.float32)),
            'cloud': torch.from_numpy(cloud.astype(np.float32)),

            # normalization
            'extent': torch.from_numpy(extent),  # [3]
            'lfborder': torch.from_numpy(lf_border),  # [3]

            # gt of output
            # pixel gt
            'mask': torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0),
            'multi_cls_mask': torch.from_numpy(multi_cls_mask).long(),
            'region': torch.from_numpy(region * mask).long(),
            'xyz': torch.from_numpy((xyz_masked * mask[:, :, None]).astype(np.float32)).permute(2, 0, 1),
            'normal': torch.from_numpy((normal_masked * mask[:, :, None]).astype(np.float32)).permute(2, 0, 1),

            # pose gt
            'target': torch.from_numpy(target.astype(np.float32)),  # [Np, 3]
            'target_r': torch.from_numpy(target_r.astype(np.float32)),
            'target_t': torch.from_numpy(target_t.astype(np.float32)),

            # data for viz, commet when not use
            'depth': torch.from_numpy(depth.astype(np.float32)),
            'mask_obj': torch.from_numpy(mask_obj.astype(np.float32)).unsqueeze(dim=0),
            'mask_label': torch.from_numpy(mask_label.astype(np.float32)).unsqueeze(dim=0),
            'mask_depth': torch.from_numpy(mask_depth.astype(np.float32)).unsqueeze(dim=0),
        }

    def __getitem__(self, index):
        item = self.all_lst[index]
        ds = None
        while ds is None:
            if self.cfg.Data.RESIZE:
                ds = self._load_resize_data(item)
            else:
                ds = self._load_data(item)
            item = random.choice(self.all_lst)

        if self.ori_image.any():
            ds['ori_img'] = self.ori_image

        return ds

    def __len__(self):
        return len(self.all_lst)

    @property
    def _error_data(self):
        cc = torch.tensor([0.])
        return {'img': cc, 'flag': False}

    def _load_model(self, model_path):
        num_pt_mesh = self.num_pt_sample
        import time
        st = time.time()
        model_points = ply_vtx(model_path) / self.scale

        xmax, xmin = model_points[:, 0].max(), model_points[:, 0].min()
        ymax, ymin = model_points[:, 1].max(), model_points[:, 1].min()
        zmax, zmin = model_points[:, 2].max(), model_points[:, 2].min()
        extent = np.array([xmax-xmin, ymax-ymin, zmax-zmin])
        lf_border = np.array([xmin, ymin, zmin])
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - num_pt_mesh)
        model_points = np.delete(model_points, dellist, axis=0)

        return model_points, extent, lf_border

    @staticmethod
    def rt2matrix(r, t):
        ext = np.eye(4, 4)
        ext[:3, :3] = r
        ext[:3, 3] = t
        return ext


# border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
# border_list = [-1] + list(range(32, 640, 32)) + [640]
border_list = [-1] + list(range(40, 640, 40)) + [640]


def read_lines(p):
    with open(p, 'r') as f:
        return [line.strip() for line in f.readlines()]


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
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
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def get_square_bbox(bbox, height_px=480, width_px=640):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]

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



def main():
    root_path = '/root/Source/ymy_dataset/Linemod_preprocessed'
    from numpy import array, float32
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from lib.network.torch_utils import my_colla_fn

    with open('dataset_config/fps_64.pkl', 'rb') as f:
        fps = pickle.load(f)

    dataset = PoseDataset('train', 1024, True, root_path, 0.000, 8, cls_type='all')

    # dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=10, collate_fn=my_colla_fn)
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    ds_iter = dataloader_test.__iter__()
    for _ in range(10):
        datas = ds_iter.__next__()
        print(datas['multi_cls_mask'].shape)
        print(datas['region'].shape)



if __name__ == '__main__':
    # a = torch.randn(2, 4, 3, 3)
    # print(a)
    # print(a.view(2, 4, -1))
    main()
