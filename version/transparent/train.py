#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/9 22:35
# @Author  : yaomy

# --------------------------------------------------------
# 通过croped的图像(图像不在网络中心)来进行预测，位移和旋转进行特殊处理
# --------------------------------------------------------

import os
import copy
import argparse
import time
import random

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn
import torch.distributed
import torch.multiprocessing as mp

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from mmcv import Config

from datasets.cleargrasp.dataset import BathPoseDataset as BathPoseDataset_trans
from datasets.cleargrasp.dataset import PoseDataset as PoseDataset_trans
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from lib.networks.loss import Loss, MultiLoss
from lib.networks.network import PoseNet
from lib.networks.TRPES import TRPESNet
from lib.networks.BathNetwork import BatchPoseNet
from lib.log import setup_logger


import cv2.cv2 as cv2


cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def _init_paramter():
    """ init paramter from parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cleargrasp', help='ycb or linemod or fuse or cleargrasp')
    parser.add_argument('--dataset_root', type=str, default='/root/Source/ymy_dataset/cleargrasp',
                        help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'' or ''ClearGrasp'')')
    parser.add_argument('--out_root', type=str, default='/root/Source/ymy_dataset/trained/instance/cleargrasp',
                        help='root of output')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--lr', default=0.0001, help='learning rate')
    parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
    parser.add_argument('--w', default=0.015, help='learning rate')
    parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
    parser.add_argument('--decay_margin', default=0.021, help='margin to decay lr & w')
    parser.add_argument('--refine_margin', default=0.0062, help='margin to start the training of iterative refinement')
    parser.add_argument('--noise_trans', default=0.03,
                        help='range of the random noise of translation added to the training data')
    parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
    parser.add_argument('--resume_posenet', type=str, default='', help='resume PoseNet model')
    parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to start')
    parser.add_argument('--debug', default=False, action="store_true", help='debug')
    parser.add_argument('--loss_type', type=str, default='reg', help='debug')
    parser.add_argument('--log_file', type=str, default='100', help='log file')
    parser.add_argument('--config_file', type=str, default='config/cleargrasp_config.py', help='log file')
    parser.add_argument(
        '--contiune_train', default=False, action="store_true",
        help='Start with the next epoch or start from over, if Ture start from over. '
             'Notethat this clears the directory of log!'
    )
    arg = parser.parse_args()
    cfg = Config.fromfile(arg.config_file)
    arg.loss_weight = cfg.TRAIN.LOSS_WEIGHT
    arg.outf = f'{arg.out_root}/{arg.log_file}'  # folder to save trained models
    arg.log_dir = f'{arg.out_root}/{arg.log_file}'  # folder to save logs
    arg.num_points = cfg.TRAIN.NUM_POINTS
    arg.num_objects = cfg.TRAIN.NUM_OBJECTS
    arg.decay_start = False
    arg.epoch_step = cfg.TRAIN.EPOCH_STEP
    arg.TRAIN = cfg.TRAIN
    if arg.debug:
        arg.workers = 0
    return arg


def worker_init_fn(worker_id):
    """固定每一次epoch的顺序，以及每个epoch的顺序，但是各个epoch之间不相同
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def seed_torch(seed=1029):
    """ Set fixed random seeds
    """
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # 固定随机算法，降低计算效率，但是没有随机性
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # 使用随机算法，计算效率高, 默认设置是这些
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


class Trainer:
    r""" Reasonably generic trainer for pytorch models
    """
    def __init__(
            self,
            model: nn.Module,
            measure,
            optimizer_function,
            paramters,
            viz=None,
            checkpoint=None,
            lr_scheduler=None,
    ):

        self.opt, self.criterion, self.model, self.checkpoint, self.optimizer_function, self.viz = (
            paramters,
            measure,
            model,
            checkpoint,
            optimizer_function,
            viz
        )

        self.lr_scheduler = lr_scheduler
        self.best_test = np.Inf

        if self.checkpoint:
            # load check point
            print('Discover the pre-training model', self.checkpoint)
            self._load_checkpoint()
        self.global_step = self.opt.start_epoch*5000
        self.optimizer = self.optimizer_function(model.parameters(), lr=self.opt.lr)
        self.st_time = time.time()

    def _load_checkpoint(self):
        if 'current' in self.opt.resume_posenet:
            self.model.load_state_dict(torch.load(self.checkpoint, map_location=torch.device('cpu')))
        else:
            stem = str(os.path.join(self.opt.resume_posenet).split('/')[-1])
            epoch = stem.split('_')[-2]
            if not self.opt.contiune_train:
                self.opt.start_epoch = int(epoch) + 1
            self.model.load_state_dict(torch.load(self.checkpoint, map_location=torch.device('cpu')))
            current_dis = float(stem.split('_')[-1].split('.pth')[0])
            self.best_test = current_dis

            if current_dis < self.opt.decay_margin:
                self.opt.decay_start = True
                self.opt.lr *= self.opt.lr_rate
                self.opt.w *= self.opt.w_rate

    @staticmethod
    def _batch_sum(a, b):
        """sum two dicts
        """
        return {k: v+b[k] for k, v in a.items()}

    def train(self, train_dataloader, test_dataloader, ds):
        for epoch in range(self.opt.start_epoch, self.opt.nepoch):
            # Reset numpy seed.
            # REF: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed()
            self._train_ephoch(train_dataloader, test_dataloader, epoch, ds)

    def _train_ephoch(self, train_dataloader, test_dataloader, epoch, ds):
        loss_dict_avg = self._loss_dict_average
        train_count = 0
        batch_count = 0
        self.model.train()
        self.optimizer.zero_grad()
        suc_dict = ds.suc_dict
        num_dict = ds.num_dict
        obj_name = ds.res_name

        logger = setup_logger('epoch%d' % epoch, os.path.join(self.opt.log_dir, 'epoch_%d_log.txt' % epoch),
                              debug=self.opt.debug)
        # train
        for i, datas in enumerate(train_dataloader, 0):
            if not datas['flag'].item():
                continue

            if batch_count >= self.opt.epoch_step:
                break

            for index in range(len(datas['img_cropeds'])):
                pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask, pred_boundry, choose = self.model(
                    Variable(datas['img_cropeds'][index]).cuda(),
                    Variable(datas['intrinsic'][index]).cuda(),
                    Variable(datas['xmaps'][index]).cuda(),
                    Variable(datas['ymaps'][index]).cuda(),
                    Variable(datas['d_scales'][index]).cuda(),
                    Variable(datas['obj_ids'][index]).cuda(),
                )

                loss, loss_dict = self.criterion(
                    pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask, pred_boundry, choose,
                    target=datas['targets'][index].cuda(),
                    model_points=datas['model_points'][index].cuda(),
                    idx=datas['obj_ids'][index].cuda(),
                    w=self.opt.w,
                    gt_n=datas['normals'][index].cuda(),
                    gt_d=datas['depths'][index].cuda(),
                    gt_m=datas['masks'][index].cuda(),
                    axis=datas['axis'][index].cuda(),
                    gt_r=datas['target_rs'][index].cuda(),
                    gt_b=datas['boundary'][index].cuda()
                )
                loss.backward()

                loss_dict_avg = self._batch_sum(loss_dict_avg, loss_dict)

                train_count += 1

                if train_count % self.opt.batch_size == 0:
                    self.global_step += 1

                    loss_dict_avg = {k: v/self.opt.batch_size for k, v in loss_dict_avg.items()}

                    for key, l in loss_dict.items():
                        self.viz.add_scalar(key, l.item(), self.global_step)

                    self.viz.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.global_step)

                    batch_count += 1
                    logger.info(
                        f'Train time {self._time} '
                        f'Epoch {epoch} Batch {int(batch_count)} Frame {train_count} '
                        f'Avg_dis:{loss_dict_avg["distance"]:.06f} '
                        f'Loss_add:{loss_dict_avg["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                        f'Loss_r:{loss_dict_avg["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                        f'Loss_normal:{loss_dict_avg["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                        f'Loss_depth:{loss_dict_avg["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                        f'Loss_mask:{loss_dict_avg["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                        f'Loss_boundary:{loss_dict_avg["loss_b"]*self.opt.loss_weight["boundary"]:.04f} '
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # reset
                    loss_dict_avg = self._loss_dict_average

                if train_count != 0 and train_count % 1000 == 0:
                    torch.save(self.model.state_dict(), '{0}/pose_model_current.pth'.format(self.opt.outf))
                self.viz.flush()

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger(
            'epoch%d_test' % epoch, os.path.join(self.opt.log_dir,  'epoch_%d_test_log.txt' % epoch),
            debug=self.opt.debug
        )

        logger.info(f'Test time {self._time}, Testing started')
        self.model.eval()
        test_dis = 0.0
        test_loss_m = 0.0
        test_count = 0
        succ = 0
        # test
        for j, datas in enumerate(test_dataloader, 0):
            if not datas['flag'].item():
                continue
            for index in range(len(datas['img_cropeds'])):
                with torch.no_grad():
                    pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask, pred_boundry, choose = self.model(
                        datas['img_cropeds'][index].cuda(),
                        datas['intrinsic'][index].cuda(),
                        datas['xmaps'][index].cuda(),
                        datas['ymaps'][index].cuda(),
                        datas['d_scales'][index].cuda(),
                        datas['obj_ids'][index].cuda()
                    )

                    loss, loss_dict = self.criterion(
                        pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask, pred_boundry, choose,
                        target=datas['targets'][index].cuda(),
                        model_points=datas['model_points'][index].cuda(),
                        idx=datas['obj_ids'][index].cuda(),
                        w=self.opt.w,
                        gt_n=datas['normals'][index].cuda(),
                        gt_d=datas['depths'][index].cuda(),
                        gt_m=datas['masks'][index].cuda(),
                        axis=datas['axis'][index].cuda(),
                        gt_r=datas['target_rs'][index].cuda(),
                        gt_b=datas['boundary'][index].cuda()
                    )

                test_dis += copy.copy(loss_dict["distance"].item())
                test_loss_m += copy.copy(loss_dict["loss_m"].item())
                test_count += 1

                if test_count in [200, 400, 600, 800]:
                    # print(pred_n.shape, gt_normal.shape)
                    self._viz_pred(
                        test_count, epoch,
                        normal=pred_normal[0], depth=pred_depth[0], mask=pred_mask[0],
                        normal_gt=datas['normals'][index][0], depth_gt=datas['depths'][index][0],
                        mask_gt=datas['masks'][index][0]
                    )
                if self.opt.dataset == 'cleargrasp':
                    metric_line = 0.1 * self.opt.diameter[datas['obj_ids'][index].item()]
                else:
                    metric_line = 0.02

                if loss_dict['distance'].item() <= metric_line:
                    suc_dict[str(datas['obj_ids'][index].item())] += 1
                    logger.info(
                        f'Pass, Test time {self._time} '
                        f'Epoch {epoch}  Test Frame No.{test_count} '
                        f'Avg_dis:{loss_dict["distance"]:.06f} '
                        f'Loss_add:{loss_dict["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                        f'Loss_r:{loss_dict["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                        f'Loss_normal:{loss_dict["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                        f'Loss_depth:{loss_dict["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                        f'Loss_mask:{loss_dict["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                        f'Loss_boundary:{loss_dict["loss_b"]*self.opt.loss_weight["boundary"]:.04f} '
                    )
                    succ += 1
                else:
                    logger.info(
                        f'NOT PASS, Test {self._time} '
                        f'Epoch {epoch} Test Frame No.{test_count} '
                        f'Avg_dis:{loss_dict["distance"]:.06f} '
                        f'Loss_add:{loss_dict["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                        f'Loss_r:{loss_dict["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                        f'Loss_normal:{loss_dict["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                        f'Loss_depth:{loss_dict["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                        f'Loss_mask:{loss_dict["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                        f'Loss_boundary:{loss_dict["loss_b"]*self.opt.loss_weight["boundary"]:.04f} '
                    )
                num_dict[str(datas['obj_ids'][index].item())] += 1
                self.viz.flush()

        test_dis = test_dis / test_count
        test_loss_m = test_loss_m / test_count
        if test_loss_m < 0.013:
            self.model.rand_n = False
        for k in num_dict.keys():
            logger.info(
                f'{obj_name[k]}:{suc_dict[k]/num_dict[k]}'
            )

        logger.info(
            f'Test time {self._time} Epoch {epoch} TEST FINISH Avg dis: {test_dis} succ: {succ / test_count}'
        )
        if test_dis <= self.best_test:
            self.best_test = test_dis
            torch.save(self.model.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if self.best_test < self.opt.decay_margin and not self.opt.decay_start:
            self.opt.decay_start = True
            self.opt.lr *= self.opt.lr_rate
            self.opt.w *= self.opt.w_rate
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr)

    def _viz_pred(self, figure, epoch, **kwargs):
        imgs = []
        if 'normal' in kwargs.keys():
            pred = kwargs['normal']
            gt = kwargs['normal_gt']

            viz_n = torch.vstack([
                ((pred * 0.5 + 0.5) * 255.).permute((1, 2, 0)).detach().cpu(),
                ((gt * 0.5 + 0.5) * 255.).permute((1, 2, 0)).detach().cpu()
            ]).numpy().astype(np.uint8)
            imgs.append(viz_n)

        if 'depth' in kwargs.keys():
            pred = kwargs['depth']
            gt = kwargs['depth_gt']
            viz_d = torch.vstack([
                pred.permute((1, 2, 0)).detach().cpu(),
                gt.permute((1, 2, 0)).detach().cpu()
            ]).repeat(1, 1, 3).numpy().astype(np.uint8)
            imgs.append(viz_d)

        if 'mask' in kwargs.keys():
            pred = kwargs['mask']
            gt = kwargs['mask_gt']
            viz_m = torch.vstack([
                (pred*255.).permute((1, 2, 0)).detach().cpu(),
                (gt*255.).permute((1, 2, 0)).detach().cpu()
            ]).repeat(1, 1, 3).numpy().astype(np.uint8)
            imgs.append(viz_m)

        img = np.hstack(imgs)
        self.viz.add_image(f'Picture: {figure}', img.transpose((2, 0, 1)), epoch)

    @property
    def _time(self):
        """ The time since training or testing began
        """
        return time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.st_time))

    @property
    def _loss_dict_average(self):
        return {
            'all_loss': 0,
            'distance': 0,
            'loss_add': 0,
            'loss_r': 0,
            'loss_n': 0,
            'loss_m': 0,
            'loss_d': 0,
            'loss_b': 0,
        }


class TrainerUnet(Trainer):
    def __init__(
            self,
            model: nn.Module,
            measure,
            optimizer_function,
            paramters,
            viz=None,
            checkpoint=None,
            lr_scheduler=None,
    ):
        super(TrainerUnet, self).__init__(
            model,
            measure,
            optimizer_function,
            paramters,
            viz,
            checkpoint,
            lr_scheduler,
        )

    def _train_ephoch(self, train_dataloader, test_dataloader, epoch, ds):
        train_count = 0
        batch_count = 0
        self.model.train()
        self.optimizer.zero_grad()
        suc_dict = ds.suc_dict
        num_dict = ds.num_dict
        obj_name = ds.res_name
        logger = setup_logger('epoch%d' % epoch, os.path.join(self.opt.log_dir, 'epoch_%d_log.txt' % epoch),
                              debug=self.opt.debug)

        for i, datas in enumerate(train_dataloader, 0):
            cc = torch.where(datas['flag'])[0]
            if len(cc) != self.opt.batch_size:
                print('Error data, Continue!')
                continue

            if batch_count >= self.opt.epoch_step:
                print('Finished one epoch')
                break

            pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask = self.model(
                Variable(datas['img_cropeds']).cuda(),
                Variable(datas['intrinsic']).cuda(),
                Variable(datas['xmaps']).cuda(),
                Variable(datas['ymaps']).cuda(),
                Variable(datas['d_scales']).cuda(),
                Variable(datas['obj_ids']).cuda(),
            )

            loss, loss_dict = self.criterion(
                pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask,
                targets=datas['targets'].cuda(),
                model_points=datas['model_points'].cuda(),
                idxs=datas['obj_ids'].cuda(),
                w=self.opt.w,
                gt_ns=datas['normals'].cuda(),
                gt_ds=datas['depths'].cuda(),
                gt_ms=datas['masks'].cuda(),
                axises=datas['axis'].cuda(),
                gt_rs=datas['target_rs'].cuda(),
            )

            if not torch.isfinite(loss):
                logger.info('INF Loss, PASS!')
                continue
            loss.backward()

            # tensorboard 可视化
            self.global_step += 1
            batch_count += 1
            for key, l in loss_dict.items():
                self.viz.add_scalar(key, l.item(), self.global_step)
            self.viz.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.global_step)
            self.viz.flush()
            logger.info(
                f'Train time {self._time} '
                f'Epoch {epoch} Batch {int(batch_count)} Frame {train_count} '
                f'Avg_dis:{loss_dict["distance"]:.06f} '
                f'Loss_add:{loss_dict["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                f'Loss_r:{loss_dict["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                f'Loss_normal:{loss_dict["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                f'Loss_depth:{loss_dict["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                f'Loss_mask:{loss_dict["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            if train_count != 0 and train_count % 1000 == 0:
                torch.save(self.model.state_dict(), '{0}/pose_model_current.pth'.format(self.opt.outf))
        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

        logger = setup_logger(
            'epoch%d_test' % epoch, os.path.join(self.opt.log_dir, 'epoch_%d_test_log.txt' % epoch),
            debug=self.opt.debug
        )
        logger.info(f'Test time {self._time}, Testing started')
        self.model.eval()
        test_dis = 0.0
        test_count = 0
        succ = 0
        for j, datas in enumerate(test_dataloader, 0):
            cc = torch.where(datas['flag'])[0]
            if len(cc) != self.opt.batch_size:
                print('Error data, Continue!')
                continue
            with torch.no_grad():
                pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask = self.model(
                    Variable(datas['img_cropeds']).cuda(),
                    Variable(datas['intrinsic']).cuda(),
                    Variable(datas['xmaps']).cuda(),
                    Variable(datas['ymaps']).cuda(),
                    Variable(datas['d_scales']).cuda(),
                    Variable(datas['obj_ids']).cuda(),
                )

                loss, loss_dict = self.criterion(
                    pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask,
                    targets=datas['targets'].cuda(),
                    model_points=datas['model_points'].cuda(),
                    idxs=datas['obj_ids'].cuda(),
                    w=self.opt.w,
                    gt_ns=datas['normals'].cuda(),
                    gt_ds=datas['depths'].cuda(),
                    gt_ms=datas['masks'].cuda(),
                    axises=datas['axis'].cuda(),
                    gt_rs=datas['target_rs'].cuda(),
                )

            test_dis += copy.copy(loss_dict["distance"].item())
            test_count += 1

            if test_count in [200, 400, 600, 800]:
                # print(pred_n.shape, gt_normal.shape)
                self._viz_pred(
                    test_count, epoch,
                    normal=pred_normal[0], depth=pred_depth[0], mask=pred_mask[0],
                    normal_gt=datas['normals'][0], depth_gt=datas['depths'][0],
                    mask_gt=datas['masks'][0]
                )

            if self.opt.dataset == 'cleargrasp':
                metric_line = 0.1 * self.opt.diameter[datas['obj_ids'].item()]
            else:
                metric_line = 0.02
            if loss_dict['distance'].item() <= metric_line:
                suc_dict[str(datas['obj_ids'].item())] += 1
                logger.info(
                    f'Pass, Test time {self._time} '
                    f'Epoch {epoch}  Test Frame No.{test_count} '
                    f'Avg_dis:{loss_dict["distance"]:.06f} '
                    f'Loss_add:{loss_dict["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                    f'Loss_r:{loss_dict["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                    f'Loss_normal:{loss_dict["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                    f'Loss_depth:{loss_dict["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                    f'Loss_mask:{loss_dict["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                )
                succ += 1
            else:
                logger.info(
                    f'NOT PASS, Test {self._time} '
                    f'Epoch {epoch} Test Frame No.{test_count} '
                    f'Avg_dis:{loss_dict["distance"]:.06f} '
                    f'Loss_add:{loss_dict["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                    f'Loss_r:{loss_dict["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                    f'Loss_normal:{loss_dict["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                    f'Loss_depth:{loss_dict["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                    f'Loss_mask:{loss_dict["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                )
            num_dict[str(datas['obj_ids'].item())] += 1
            self.viz.flush()

        test_dis = test_dis / test_count
        for k in num_dict.keys():
            logger.info(
                f'{obj_name[k]}:{suc_dict[k]/num_dict[k]}'
            )

        logger.info(
            f'Test time {self._time} Epoch {epoch} TEST FINISH, Avg dis: {test_dis} Succ: {succ / test_count}'
        )
        if test_dis <= self.best_test:
            self.best_test = test_dis
            torch.save(self.model.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if self.best_test < self.opt.decay_margin and not self.opt.decay_start:
            self.opt.decay_start = True
            self.opt.lr *= self.opt.lr_rate
            self.opt.w *= self.opt.w_rate
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr)

    @property
    def _loss_dict_average(self):
        return {k: 0. for k in self.criterion.loss_dict.keys()}


class TrainerDistributed(Trainer):
    def __init__(
            self,
            model: nn.Module,
            measure,
            optimizer_function,
            paramters,
            viz=None,
            checkpoint=None,
            lr_scheduler=None,
            rank=None
    ):
        super(TrainerDistributed, self).__init__(
            model,
            measure,
            optimizer_function,
            paramters,
            viz,
            None,
            lr_scheduler,
        )
        self.checkpoint = checkpoint
        if self.checkpoint:
            # load check point
            print('Discover the pre-training model', self.checkpoint)
            self._load_checkpoint()

        self.rank = rank
        self.opt.epoch_step = int(self.opt.TRAIN.EPOCH_STEP/self.opt.nprocs)

    # def _load_checkpoint(self):
    #     if 'current' in self.opt.resume_posenet:
    #         self.model.load_state_dict(torch.load({
    #             k.replace('module.', ''): v for k, v in self.checkpoint['state_dict'].items()
    #         }, map_location=torch.device('cpu')))
    #     else:
    #         stem = str(os.path.join(self.opt.resume_posenet).split('/')[-1])
    #         epoch = stem.split('_')[-2]
    #         if not self.opt.contiune_train:
    #             self.opt.start_epoch = int(epoch) + 1
    #         self.model.load_state_dict(torch.load({
    #             k.replace('module.', ''): v for k, v in self.checkpoint['state_dict'].items()
    #         }, map_location=torch.device('cpu')))
    #         current_dis = float(stem.split('_')[-1].split('.pth')[0])
    #         self.best_test = current_dis

    #         if current_dis < self.opt.decay_margin:
    #             self.opt.decay_start = True
    #             self.opt.lr *= self.opt.lr_rate
    #             self.opt.w *= self.opt.w_rate

    def train_epoch(self, train_dataloader, epoch, logger):
        batch_count = 0
        self.model.train()
        self.optimizer.zero_grad()
        for i, datas in enumerate(train_dataloader, 0):
            cc = torch.where(datas['flag'])[0]
            if len(cc) != self.opt.batch_size:
                print('Error data, Continue!')
                continue

            # 中止条件
            # if batch_count >= self.opt.epoch_step:
            #     # if self.rank == 0:
            #     #     print('Finished one epoch')
            #     break

            pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask = self.model(
                Variable(datas['img_cropeds']).cuda(),
                Variable(datas['intrinsic']).cuda(),
                Variable(datas['xmaps']).cuda(),
                Variable(datas['ymaps']).cuda(),
                Variable(datas['d_scales']).cuda(),
                Variable(datas['obj_ids']).cuda(),
            )

            loss, loss_dict = self.criterion(
                pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask,
                targets=datas['targets'].cuda(),
                model_points=datas['model_points'].cuda(),
                idxs=datas['obj_ids'].cuda(),
                w=self.opt.w,
                gt_ns=datas['normals'].cuda(),
                gt_ds=datas['depths'].cuda(),
                gt_ms=datas['masks'].cuda(),
                axises=datas['axis'].cuda(),
                gt_rs=datas['target_rs'].cuda(),
            )

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.global_step += 1
            batch_count += 1
            if self.rank == 0:
                logger.info(
                    f'Train time {self._time} '
                    f'Epoch {epoch} Batch {int(batch_count)} Frame {batch_count*self.opt.batch_size} '
                    f'Avg_dis:{loss_dict["distance"]:.06f} '
                    f'Loss_add:{loss_dict["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                    f'Loss_r:{loss_dict["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                    f'Loss_normal:{loss_dict["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                    f'Loss_depth:{loss_dict["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                    f'Loss_mask:{loss_dict["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                )

                for key, l in loss_dict.items():
                    self.viz.add_scalar(key, l.item(), self.global_step)
                self.viz.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.global_step)
                self.viz.flush()
                if batch_count != 0 and batch_count % 1000 == 0:
                    torch.save(self.model.state_dict(), '{0}/pose_model_current.pth'.format(self.opt.outf))

    def test_epoch(self, test_dataloader, test_epoch, test_logger, ds):
        self.model.eval()
        test_dis = 0.0
        test_count = 0
        succ = 0
        suc_dict = ds.suc_dict
        num_dict = ds.num_dict
        obj_name = ds.res_name
        if self.rank == 0:
            test_logger.info(f'Test time {self._time}, Testing started')
        for j, datas in enumerate(test_dataloader, 0):
            cc = torch.where(datas['flag'])[0]
            if len(cc) != 1:
                print('Error data, Continue!')
                continue
            with torch.no_grad():
                pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask = self.model(
                    Variable(datas['img_cropeds']).cuda(),
                    Variable(datas['intrinsic']).cuda(),
                    Variable(datas['xmaps']).cuda(),
                    Variable(datas['ymaps']).cuda(),
                    Variable(datas['d_scales']).cuda(),
                    Variable(datas['obj_ids']).cuda(),
                )

            loss, loss_dict = self.criterion(
                pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask,
                targets=datas['targets'].cuda(),
                model_points=datas['model_points'].cuda(),
                idxs=datas['obj_ids'].cuda(),
                w=self.opt.w,
                gt_ns=datas['normals'].cuda(),
                gt_ds=datas['depths'].cuda(),
                gt_ms=datas['masks'].cuda(),
                axises=datas['axis'].cuda(),
                gt_rs=datas['target_rs'].cuda(),
            )
            test_dis += copy.copy(loss_dict["distance"].item())
            test_count += 1
            if self.rank == 0:
                if test_count in [200, 400, 600, 800]:
                    # print(pred_n.shape, gt_normal.shape)
                    self._viz_pred(
                        test_count, test_epoch,
                        normal=pred_normal[0], depth=pred_depth[0], mask=pred_mask[0],
                        normal_gt=datas['normals'][0], depth_gt=datas['depths'][0],
                        mask_gt=datas['masks'][0]
                    )
                    self.viz.flush()

            if self.opt.dataset == 'cleargrasp':
                metric_line = 0.1 * self.opt.diameter[datas['obj_ids'].item()]
            else:
                metric_line = 0.02
            num_dict[str(datas['obj_ids'].item())] += 1
            if self.rank == 0:
                if loss_dict['distance'].item() <= metric_line:
                    suc_dict[str(datas['obj_ids'].item())] += 1
                    test_logger.info(
                        f'Pass, Test time {self._time} '
                        f'Epoch {test_epoch}  Test Frame No.{test_count} '
                        f'Avg_dis:{loss_dict["distance"]:.06f} '
                        f'Loss_add:{loss_dict["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                        f'Loss_r:{loss_dict["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                        f'Loss_normal:{loss_dict["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                        f'Loss_depth:{loss_dict["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                        f'Loss_mask:{loss_dict["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                    )
                    succ += 1
                else:
                    test_logger.info(
                        f'NOT PASS, Test {self._time} '
                        f'Epoch {test_epoch} Test Frame No.{test_count} '
                        f'Avg_dis:{loss_dict["distance"]:.06f} '
                        f'Loss_add:{loss_dict["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                        f'Loss_r:{loss_dict["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                        f'Loss_normal:{loss_dict["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                        f'Loss_depth:{loss_dict["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                        f'Loss_mask:{loss_dict["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                    )

        test_dis = test_dis / test_count

        if self.rank == 0:
            for k in num_dict.keys():
                test_logger.info(
                    f'{obj_name[k]}:{suc_dict[k] / num_dict[k]}'
                )
            test_logger.info(
                f'Test time {self._time} Epoch {test_epoch} TEST FINISH, Avg dis: {test_dis} Succ: {succ / test_count}'
            )

        if test_dis <= self.best_test and self.rank == 0:
            self.best_test = test_dis
            torch.save(self.model.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, test_epoch, test_dis))
            print(test_epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if self.best_test < self.opt.decay_margin and not self.opt.decay_start and self.rank == 0:
            self.opt.decay_start = True
            self.opt.lr *= self.opt.lr_rate
            self.opt.w *= self.opt.w_rate
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr)


class TrainerDistributedPerBath(Trainer):
    def __init__(
            self,
            model: nn.Module,
            measure,
            optimizer_function,
            paramters,
            viz=None,
            checkpoint=None,
            lr_scheduler=None,
            rank=None
    ):
        super(TrainerDistributedPerBath, self).__init__(
            model,
            measure,
            optimizer_function,
            paramters,
            viz,
            None,
            lr_scheduler,
        )
        self.checkpoint = checkpoint
        if self.checkpoint:
            # load check point
            print('Discover the pre-training model', self.checkpoint)
            self._load_checkpoint()
        self.rank = rank
        self.opt.epoch_step = int(self.opt.TRAIN.EPOCH_STEP/self.opt.nprocs)

    # def _load_checkpoint(self):
    #     if 'current' in self.opt.resume_posenet:
    #         self.model.load_state_dict(torch.load(self.checkpoint, map_location=torch.device('cpu')))
    #     else:
    #         stem = str(os.path.join(self.opt.resume_posenet).split('/')[-1])
    #         epoch = stem.split('_')[-2]
    #         if not self.opt.contiune_train:
    #             self.opt.start_epoch = int(epoch) + 1
    #         self.model.load_state_dict(torch.load({
    #             k.replace('module.', ''): v for k, v in self.checkpoint['state_dict'].items()
    #         }, map_location=torch.device('cpu')))
    #         current_dis = float(stem.split('_')[-1].split('.pth')[0])
    #         self.best_test = current_dis

    #         if current_dis < self.opt.decay_margin:
    #             self.opt.decay_start = True
    #             self.opt.lr *= self.opt.lr_rate
    #             self.opt.w *= self.opt.w_rate

    def train_epoch(self, train_dataloader, epoch, logger):
        train_count = 0
        batch_count = 0
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict_avg = self._loss_dict_average
        for i, datas in enumerate(train_dataloader, 0):

            if not datas['flag'].item():
                print('Error data, Continue!')
                print(datas['paths'])
                continue
            # 中止条件
            if batch_count >= self.opt.epoch_step:
                break

            for index in range(len(datas['img_cropeds'])):
                pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask = self.model(
                    Variable(datas['img_cropeds'][index]).cuda(),
                    Variable(datas['intrinsic'][index]).cuda(),
                    Variable(datas['xmaps'][index]).cuda(),
                    Variable(datas['ymaps'][index]).cuda(),
                    Variable(datas['d_scales'][index]).cuda(),
                    Variable(datas['obj_ids'][index]).cuda(),
                )

                loss, loss_dict = self.criterion(
                    pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask,
                    targets=datas['targets'][index].cuda(),
                    model_points=datas['model_points'][index].cuda(),
                    idxs=datas['obj_ids'][index].cuda(),
                    w=self.opt.w,
                    gt_ns=datas['normals'][index].cuda(),
                    gt_ds=datas['depths'][index].cuda(),
                    gt_ms=datas['masks'][index].cuda(),
                    axises=datas['axis'][index].cuda(),
                    gt_rs=datas['target_rs'][index].cuda(),
                )
                loss.backward()
                train_count += 1
                loss_dict_avg = self._batch_sum(loss_dict_avg, loss_dict)

                if train_count % self.opt.batch_size == 0:
                    self.global_step += 1
                    batch_count += 1
                    loss_dict_avg = {k: v/self.opt.batch_size for k, v in loss_dict_avg.items()}
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # reset
                    if self.rank == 0:
                        for key, l in loss_dict.items():
                            self.viz.add_scalar(key, l.item(), self.global_step)
                        self.viz.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'],
                                            self.global_step)
                        logger.info(
                            f'Train time {self._time} '
                            f'Epoch {epoch} Batch {int(batch_count)} Frame {train_count} '
                            f'Avg_dis:{loss_dict_avg["distance"]:.06f} '
                            f'Loss_add:{loss_dict_avg["loss_add"]*self.opt.loss_weight["distance"]:.04f} '
                            f'Loss_r:{loss_dict_avg["loss_r"]*self.opt.loss_weight["rotation"]:.04f} '
                            f'Loss_normal:{loss_dict_avg["loss_n"]*self.opt.loss_weight["normal"]:.04f} '
                            f'Loss_depth:{loss_dict_avg["loss_d"]*self.opt.loss_weight["depth"]:.04f} '
                            f'Loss_mask:{loss_dict_avg["loss_m"]*self.opt.loss_weight["mask"]:.04f} '
                        )
                        self.viz.flush()
                        if batch_count != 0 and batch_count % 1000 == 0:
                            torch.save(self.model.state_dict(), '{0}/pose_model_current.pth'.format(self.opt.outf))
                        loss_dict_avg = self._loss_dict_average

    def test_epoch(self, test_dataloader, test_epoch, test_logger, ds):
        self.model.eval()
        test_dis = 0.0
        test_count = 0
        succ = 0
        suc_dict = ds.suc_dict
        num_dict = ds.num_dict
        obj_name = ds.res_name
        if self.rank == 0:
            test_logger.info(f'Test time {self._time}, Testing started')
        for j, datas in enumerate(test_dataloader, 0):
            if not datas['flag'].item():
                continue
            for index in range(len(datas['img_cropeds'])):
                with torch.no_grad():
                    pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask = self.model(
                        Variable(datas['img_cropeds'][index]).cuda(),
                        Variable(datas['intrinsic'][index]).cuda(),
                        Variable(datas['xmaps'][index]).cuda(),
                        Variable(datas['ymaps'][index]).cuda(),
                        Variable(datas['d_scales'][index]).cuda(),
                        Variable(datas['obj_ids'][index]).cuda(),
                    )

                    loss, loss_dict = self.criterion(
                        pred_r, pred_t, pred_c, pred_normal, pred_depth, pred_mask,
                        targets=datas['targets'][index].cuda(),
                        model_points=datas['model_points'][index].cuda(),
                        idxs=datas['obj_ids'][index].cuda(),
                        w=self.opt.w,
                        gt_ns=datas['normals'][index].cuda(),
                        gt_ds=datas['depths'][index].cuda(),
                        gt_ms=datas['masks'][index].cuda(),
                        axises=datas['axis'][index].cuda(),
                        gt_rs=datas['target_rs'][index].cuda(),
                    )

                test_dis += copy.copy(loss_dict["distance"].item())
                test_count += 1
                if self.rank == 0:
                    if test_count in [200, 400, 600, 800]:
                        # print(pred_n.shape, gt_normal.shape)
                        self._viz_pred(
                            test_count, test_epoch,
                            normal=pred_normal[0], depth=pred_depth[0], mask=pred_mask[0],
                            normal_gt=datas['normals'][index][0], depth_gt=datas['depths'][index][0],
                            mask_gt=datas['masks'][index][0]
                        )
                        self.viz.flush()
                if self.opt.dataset == 'cleargrasp':
                    metric_line = 0.1 * self.opt.diameter[datas['obj_ids'][index].item()]
                else:
                    metric_line = 0.02
                num_dict[str(datas['obj_ids'][index].item())] += 1

                if self.rank == 0:
                    if loss_dict['distance'].item() <= metric_line:
                        suc_dict[str(datas['obj_ids'][index].item())] += 1
                        test_logger.info(
                            f'Pass, Test time {self._time} '
                            f'Epoch {test_epoch}  Test Frame No.{test_count} '
                            f'Avg_dis:{loss_dict["distance"]:.06f} '
                            f'Loss_add:{loss_dict["loss_add"] * self.opt.loss_weight["distance"]:.04f} '
                            f'Loss_r:{loss_dict["loss_r"] * self.opt.loss_weight["rotation"]:.04f} '
                            f'Loss_normal:{loss_dict["loss_n"] * self.opt.loss_weight["normal"]:.04f} '
                            f'Loss_depth:{loss_dict["loss_d"] * self.opt.loss_weight["depth"]:.04f} '
                            f'Loss_mask:{loss_dict["loss_m"] * self.opt.loss_weight["mask"]:.04f} '
                        )
                        succ += 1
                    else:
                        test_logger.info(
                            f'NOT PASS, Test {self._time} '
                            f'Epoch {test_epoch} Test Frame No.{test_count} '
                            f'Avg_dis:{loss_dict["distance"]:.06f} '
                            f'Loss_add:{loss_dict["loss_add"] * self.opt.loss_weight["distance"]:.04f} '
                            f'Loss_r:{loss_dict["loss_r"] * self.opt.loss_weight["rotation"]:.04f} '
                            f'Loss_normal:{loss_dict["loss_n"] * self.opt.loss_weight["normal"]:.04f} '
                            f'Loss_depth:{loss_dict["loss_d"] * self.opt.loss_weight["depth"]:.04f} '
                            f'Loss_mask:{loss_dict["loss_m"] * self.opt.loss_weight["mask"]:.04f} '
                        )

        test_dis = test_dis / test_count

        if self.rank == 0:
            for k in num_dict.keys():
                test_logger.info(
                    f'{obj_name[k]}:{suc_dict[k] / num_dict[k]}'
                )
            test_logger.info(
                f'Test time {self._time} Epoch {test_epoch} TEST FINISH, Avg dis: {test_dis} Succ: {succ / test_count}'
            )

            if test_dis <= self.best_test and self.rank == 0:
                self.best_test = test_dis
                torch.save(self.model.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, test_epoch, test_dis))
                print(f'>>>>>>>>----------EPOCH {test_epoch}, BEST TEST MODEL SAVED---------<<<<<<<<')

            if self.best_test < self.opt.decay_margin and not self.opt.decay_start and self.rank == 0:
                self.opt.decay_start = True
                self.opt.lr *= self.opt.lr_rate
                self.opt.w *= self.opt.w_rate
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.opt.lr)

    @property
    def _loss_dict_average(self):
        return {
            'all_loss': 0,
            'distance': 0,
            'loss_add': 0,
            'loss_r': 0,
            'loss_n': 0,
            'loss_m': 0,
            'loss_d': 0,
        }


def train_single(opt):

    # torch.distributed.init_process_group(backend='nccl')

    print(f'Loading {opt.dataset} Dataset...')
    if opt.dataset == 'cleargrasp':
        PoseDataset = PoseDataset_trans
    elif opt.dataset == 'ycb':
        PoseDataset = PoseDataset_ycb
    else:
        raise KeyError
    # dataset
    dataset = PoseDataset('train', 1000, True, opt.dataset_root, opt.noise_trans, False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers,
                                             worker_init_fn=worker_init_fn)

    test_dataset = PoseDataset('test', 1000, False, opt.dataset_root, 0.0, False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                                 worker_init_fn=worker_init_fn)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    if opt.dataset == 'cleargrasp':
        opt.diameter = dataset.get_diameter()

    print(
        f'>>>>>>>>----------Dataset loaded!---------<<<<<<<<\n'
        f'length of the training set: {len(dataset)}\n'
        f'length of the testing set: {len(test_dataset)}\n'
        f'number of sample points on mesh: {opt.num_points_mesh}\n'
        f'symmetry object list: {opt.sym_list}'
    )

    # model
    # estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects)
    # criterion = Loss(opt.num_points_mesh, opt.sym_list, loss_weight=opt.loss_weight)

    estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects)
    criterion = Loss(opt.num_points_mesh, opt.sym_list, loss_weight=opt.loss_weight)

    estimator.cuda()
    criterion.cuda()

    # viz throught tensorboard
    writer = SummaryWriter(opt.log_dir)

    # data_iter = dataloader.__iter__()
    # while True:
    #     data = data_iter.__next__()
    #     if not data['flag'].item():
    #         continue
    #     else:
    #         break
    #
    # writer.add_graph(estimator, [
    #     data['img_cropeds'][0].cuda(),
    #     data['intrinsic'][0].cuda(),
    #     data['xmaps'][0].cuda(),
    #     data['ymaps'][0].cuda(),
    #     data['d_scales'][0].cuda(),
    #     data['obj_ids'][0].cuda(),
    # ])

    # trainer = Trainer(estimator, criterion, optim.Adam, opt, viz=writer, checkpoint=opt.resume_posenet)
    trainer = Trainer(estimator, criterion, optim.Adam, opt, viz=writer, checkpoint=opt.resume_posenet)

    trainer.train(dataloader, testdataloader, test_dataset)


def train_batch(opt):
    """多batchsize，但是不分布式训练
    """
    # torch.distributed.init_process_group(backend='nccl')

    print(f'Loading {opt.dataset} Dataset...')
    if opt.dataset == 'cleargrasp':
        PoseDataset = BathPoseDataset_trans
    elif opt.dataset == 'ycb':
        PoseDataset = PoseDataset_ycb
    else:
        raise KeyError
    # dataset
    dataset = PoseDataset('train', 1000, True, opt.dataset_root, opt.noise_trans, False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                             worker_init_fn=worker_init_fn)

    test_dataset = PoseDataset('test', 1000, False, opt.dataset_root, 0.0, False)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers,
                                                 worker_init_fn=worker_init_fn)

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    if opt.dataset == 'cleargrasp':
        opt.diameter = dataset.get_diameter()

    print(
        f'>>>>>>>>----------Dataset loaded!---------<<<<<<<<\n'
        f'length of the training set: {len(dataset)}\n'
        f'length of the testing set: {len(test_dataset)}\n'
        f'number of sample points on mesh: {opt.num_points_mesh}\n'
        f'symmetry object list: {opt.sym_list}'
    )

    # model
    # estimator = PoseNet(num_points=opt.num_points, num_obj=opt.num_objects)
    # criterion = Loss(opt.num_points_mesh, opt.sym_list, loss_weight=opt.loss_weight)

    estimator = TRPESNet(num_points=opt.num_points, num_obj=opt.num_objects)
    criterion = MultiLoss(opt.num_points_mesh, opt.sym_list, loss_weight=opt.loss_weight)

    estimator.cuda()
    criterion.cuda()

    # viz throught tensorboard
    writer = SummaryWriter(opt.log_dir)

    # data_iter = dataloader.__iter__()
    # while True:
    #     data = data_iter.__next__()
    #     if not data['flag'].item():
    #         continue
    #     else:
    #         break
    #
    # writer.add_graph(estimator, [
    #     data['img_cropeds'][0].cuda(),
    #     data['intrinsic'][0].cuda(),
    #     data['xmaps'][0].cuda(),
    #     data['ymaps'][0].cuda(),
    #     data['d_scales'][0].cuda(),
    #     data['obj_ids'][0].cuda(),
    # ])

    # trainer = Trainer(estimator, criterion, optim.Adam, opt, viz=writer, checkpoint=opt.resume_posenet)
    trainer = TrainerUnet(estimator, criterion, optim.Adam, opt, viz=writer, checkpoint=opt.resume_posenet)

    trainer.train(dataloader, testdataloader, test_dataset)


def main_worker(gpu, opt):
    """分布式训练
    """
    print("Use GPU: {} for training".format(gpu))
    print(f'Loading {opt.dataset} Dataset...')
    rank = opt.TRAIN.MULTIGPU.RANK * opt.nprocs + gpu

    torch.distributed.init_process_group(
        backend='nccl',
        init_method=opt.TRAIN.MULTIGPU.DIST_URL,
        world_size=opt.nprocs*opt.TRAIN.MULTIGPU.WORLD_SIZE,
        rank=rank,

    )

    if opt.dataset == 'cleargrasp':
        PoseDataset = BathPoseDataset_trans
    elif opt.dataset == 'ycb':
        PoseDataset = PoseDataset_ycb
    else:
        raise KeyError
    # dataset
    dataset = PoseDataset('train', 1000, True, opt.dataset_root, opt.noise_trans, False)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        sampler=sampler,
    )

    test_dataset = PoseDataset('test', 1000, False, opt.dataset_root, 0.0, False)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.workers,
    )

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    if gpu == 0:
        print(
            f'>>>>>>>>----------Dataset loaded!---------<<<<<<<<\n'
            f'length of the training set: {len(dataset)}\n'
            f'length of the testing set: {len(test_dataset)}\n'
            f'number of sample points on mesh: {opt.num_points_mesh}\n'
            f'symmetry object list: {opt.sym_list}'
        )

    if opt.dataset == 'cleargrasp':
        opt.diameter = dataset.get_diameter()

    estimator = TRPESNet(num_points=opt.num_points, num_obj=opt.num_objects)
    # estimator = BatchPoseNet(num_points=opt.num_points, num_obj=opt.num_objects)
    torch.cuda.set_device(gpu)
    estimator = estimator.cuda(gpu)
    criterion = MultiLoss(opt.num_points, opt.sym_list, loss_weight=opt.loss_weight, knn=opt.knn)
    criterion.cuda(gpu)

    if opt.TRAIN.MULTIGPU.SYNC_BN:
        estimator = nn.SyncBatchNorm.convert_sync_batchnorm(estimator)

    estimator = torch.nn.parallel.DistributedDataParallel(
        estimator,
        device_ids=[gpu],
        find_unused_parameters=True
    )

    # # viz throught tensorboard
    if rank == 0:
        writer = SummaryWriter(opt.log_dir)
    else:
        writer = None
    trainer = TrainerDistributed(estimator, criterion, optim.Adam, opt,
                                 rank=rank, viz=writer, checkpoint=opt.resume_posenet)

    for epoch in range(opt.start_epoch, opt.nepoch):
        # np.random.seed()

        # 每个epoch给sample添加一个随机种子，不加这行代码每个epoch采样出来的数据是一样的
        # https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20distributed%20distributedsampler#:~:text=replicas.%20Default%3A%20False.-,WARNING,-In%20distributed%20mode
        sampler.set_epoch(epoch)

        if rank == 0:
            logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch), debug=opt.debug)
        else:
            logger = None

        trainer.train_epoch(dataloader, epoch, logger)

        if rank == 0:
            print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))
            test_logger = setup_logger(
                'epoch%d_test' % epoch, os.path.join(opt.log_dir,  'epoch_%d_test_log.txt' % epoch), debug=opt.debug
            )

            trainer.test_epoch(testdataloader, epoch, test_logger, test_dataset)


def main_worker_one_batch(gpu, opt):
    """分布式训练，但是batchsize还是1
    """
    print("Use GPU: {} for training".format(gpu))
    print(f'Loading {opt.dataset} Dataset...')
    rank = opt.TRAIN.MULTIGPU.RANK * opt.nprocs + gpu

    torch.distributed.init_process_group(
        backend='nccl',
        init_method=opt.TRAIN.MULTIGPU.DIST_URL,
        world_size=opt.nprocs*opt.TRAIN.MULTIGPU.WORLD_SIZE,
        rank=rank,
    )

    if opt.dataset == 'cleargrasp':
        PoseDataset = PoseDataset_trans
    elif opt.dataset == 'ycb':
        PoseDataset = PoseDataset_ycb
    else:
        raise KeyError

    # dataset
    dataset = PoseDataset('train', 1000, True, opt.dataset_root, opt.noise_trans, False)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.workers,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
    )

    test_dataset = PoseDataset('test', 1000, False, opt.dataset_root, 0.0, False)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.workers,
        worker_init_fn=worker_init_fn
    )

    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()
    if gpu == 0:
        print(
            f'>>>>>>>>----------Dataset loaded!---------<<<<<<<<\n'
            f'length of the training set: {len(dataset)}\n'
            f'length of the testing set: {len(test_dataset)}\n'
            f'number of sample points on mesh: {opt.num_points_mesh}\n'
            f'symmetry object list: {opt.sym_list}'
        )

    if opt.dataset == 'cleargrasp':
        opt.diameter = dataset.get_diameter()

    estimator = TRPESNet(num_points=opt.num_points, num_obj=opt.num_objects)
    torch.cuda.set_device(gpu)
    estimator = estimator.cuda(gpu)
    criterion = MultiLoss(opt.num_points, opt.sym_list, loss_weight=opt.loss_weight, knn=opt.knn)
    criterion.cuda(gpu)

    if opt.TRAIN.MULTIGPU.SYNC_BN:
        estimator = nn.SyncBatchNorm.convert_sync_batchnorm(estimator)

    estimator = torch.nn.parallel.DistributedDataParallel(
        estimator,
        device_ids=[gpu],
        find_unused_parameters=False
    )

    # # viz throught tensorboard
    if rank == 0:
        writer = SummaryWriter(opt.log_dir)
    else:
        writer = None
    trainer = TrainerDistributedPerBath(estimator, criterion, optim.Adam, opt,
                                        rank=rank, viz=writer, checkpoint=opt.resume_posenet)

    for epoch in range(opt.start_epoch, opt.nepoch):
        # np.random.seed()

        # 每个epoch给sample添加一个随机种子，不加这行代码每个epoch采样出来的数据是一样的
        # https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20distributed%20distributedsampler#:~:text=replicas.%20Default%3A%20False.-,WARNING,-In%20distributed%20mode
        sampler.set_epoch(epoch)

        if rank == 0:
            logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch), debug=opt.debug)
        else:
            logger = None

        trainer.train_epoch(dataloader, epoch, logger)

        if rank == 0:
            print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))
            test_logger = setup_logger(
                'epoch%d_test' % epoch, os.path.join(opt.log_dir,  'epoch_%d_test_log.txt' % epoch), debug=opt.debug
            )
        else:
            test_logger = None

        trainer.test_epoch(testdataloader, epoch, test_logger, test_dataset)


def distributed_train(opt):
    # 重新编译knn
    import pykeops
    from pykeops.torch import generic_argkmin
    pykeops.clean_pykeops()
    # pykeops.test_torch_bindings()
    knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')
    A = torch.randn(500, 3)
    B = torch.randn(500, 3)
    knn(A, B)
    opt.knn = knn

    # 多进程启动器
    opt.nprocs = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        nprocs=opt.nprocs,
        args=(opt,),
    )


if __name__ == '__main__':
    # 设置固定的随机种子
    # seed_torch()
    args = _init_paramter()

    # 清空log目录
    from lib.debug import clrdir
    print('Logdir: ', args.log_dir)
    if ('trained' in args.log_dir and not args.resume_posenet) or args.contiune_train:
        clrdir(args.log_dir)

    # 修改多线程的tensor方式为file_system（默认方式为file_descriptor，受限于open files数量）
    # https://www.cnblogs.com/zhengbiqing/p/10478311.html
    torch.multiprocessing.set_sharing_strategy('file_system')

    # train_batch(args)
    distributed_train(args)
    # train_single(args)
