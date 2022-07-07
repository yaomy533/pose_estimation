#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 19:50
# @Author  : yaomy
import os
import argparse
from mmcv import Config
import cv2

import torch
import numpy as np
import torch.utils.data
import pykeops
from torch.utils.tensorboard import SummaryWriter
from pykeops.torch import generic_argkmin

from dataset.linemod.batchdataset import PoseDataset as PoseDataset_lm
from dataset.linemod.lm_bop import PoseDataset as PoseDataset_bop
from lib.utils.metric import Metric
from lib.utils.utlis import load_part_module
from lib.network.krrn import KRRN
from lib.network.loss import KRRNLoss
from lib.network.optimizer.ranger import Ranger
from tools.trainer import Trainer
from tools.trainer import Trainer as Trainer
from lib.utils.logger import setup_logger
from lib.network.torch_utils import build_lr_scheduler, my_colla_fn, worker_init_fn


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='linemod', help='ycb or linemod or fuse or cleargrasp')
parser.add_argument('--dataset_root', type=str, default='/root/Source/ymy_dataset/Linemod_preprocessed',
                    help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'' or ''ClearGrasp'')')
parser.add_argument('--out_root', type=str, default='/root/Source/ymy_dataset/trained/krrnv3',
                    help='root of output')
parser.add_argument('--noise_trans', default=0.003,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--start_epoch', type=int, default=0, help='which epoch to start')
parser.add_argument('--epoch_step', type=int, default=10000, help='how many step of one epoch')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--workers', type=int, default=20, help='number of data loading workers')
parser.add_argument('--log_file', type=str, default='01', help='log file')
parser.add_argument('--config_file', type=str, default='config/linemod/linemod_config.py', help='log file')
parser.add_argument('--eval_mode', default=False, action="store_true", help='eval mode')
parser.add_argument('--retrain', default=False, action="store_true", help='Whether to retraining the same model')
parser.add_argument('--cls_type', type=str, default='ape', help='name of cls, if None use all')
parser.add_argument('--refine', default=False, action="store_true", help='refine_start')
parser.add_argument('--backbone_oly', default=False, action="store_true", help='only load backbone')
parser.add_argument(
    '--resume_posenet', type=str,
    default='',
    help='resume PoseNet model'
)

parser.add_argument(
    '--resume_refine_posenet', type=str,
    default='',
    help='resume refine PoseNet model'
)

parser.add_argument('--debug', default=False, action="store_true", help='debug')

arg = parser.parse_args()
cfg = Config.fromfile(arg.config_file)

if arg.dataset == 'lm-bop':
    arg.dataset_root = '/root/Source/ymy_dataset/lm-bop'
    arg.train_dir = 'linemod'
    PoseDataset = PoseDataset_bop
elif arg.dataset == 'linemod':
    arg.dataset_root = '/root/Source/ymy_dataset/Linemod_preprocessed'
    arg.train_dir = 'linemod'
    PoseDataset = PoseDataset_lm

arg.log_dir = f'{arg.out_root}/{arg.train_dir}/{arg.log_file}'
arg.outf = f'{arg.out_root}/{arg.train_dir}/{arg.log_file}'
if not os.path.exists(arg.log_dir):
    os.mkdir(arg.log_dir)
print(f'log_dir: {arg.log_dir}')
arg.cfg = cfg

if arg.debug:
    print('---> debug mode')
    arg.epoch_step = 5
    arg.cfg.Train.NUM_EPOCH_REPEAT = 1

arg.decay_start = False

def train():
    print(f'Loading {arg.dataset} Dataset...')

    arg.batchsize = cfg.Train.BATCHSIZE
    arg.cfg.Train.REFINE = arg.refine
    if arg.cfg.Train.Optimizer.TYPE == 'Ranger':
        Optim = Ranger
    else:
        Optim = torch.optim.Adam

    # dataset
    dataset = PoseDataset('train', arg.cfg.Data.NUM_POINTS, False, arg.dataset_root, arg.cfg.Train.NOISE, 8, cls_type=arg.cls_type, cfg=arg.cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=arg.batchsize, shuffle=True,
                                                num_workers=arg.workers, collate_fn=my_colla_fn, worker_init_fn=worker_init_fn, pin_memory=True)

    test_dataset = PoseDataset('test', arg.cfg.Data.NUM_POINTS, False, arg.dataset_root, 0.0, 8, cls_type=arg.cls_type, cfg=arg.cfg)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=arg.workers, worker_init_fn=worker_init_fn, pin_memory=True)

    sym_list = dataset.sym_obj
    metric = Metric(sym_list)
    arg.diameter = dataset.diameter
    arg.num_cls = dataset.objlist.__len__()

    if arg.refine:
        arg.deacline_step = len(dataset) / cfg.Train.BATCHSIZE * cfg.Train.Lr.LAMBDA.DEACY_EPOCH_RFINE
    else:
        arg.deacline_step = len(dataset)/cfg.Train.BATCHSIZE * cfg.Train.Lr.LAMBDA.DEACY_EPOCH

    print(
        f'>>>>>>>>----------Dataset loaded!---------<<<<<<<<\n'
        f'length of the training set: {len(dataset)}\n'
        f'length of the testing set: {len(test_dataset)}\n'
    )

    print('Compiling KNN....')
    pykeops.clean_pykeops()
    pykeops.test_torch_bindings()
    knn = generic_argkmin('SqDist(x, y)', 'a = Vi(1)', 'x = Vi(3)', 'y = Vj(3)')
    print('Compiling KNN Success!\n')

    # base module
    # estimator = KRRN(num_kps=arg.cfg.Module.NUM_KPS, num_cls=arg.num_cls, fs=arg.cfg.Module.XYZNet.HEADEN_FS, cfg=arg.cfg)
    estimator = KRRN(num_cls=arg.num_cls, cfg=arg.cfg)
    criterion = KRRNLoss(sym_list=sym_list, knn=knn, cfg=arg.cfg)

    current_dis = np.Inf
    global_step = 0
    refine_current_dis = np.Inf

    if arg.resume_posenet:
        if 'current' not in arg.resume_posenet:
            stem = str(os.path.join(arg.resume_posenet).split('/')[-1])
            if not arg.retrain:
                if 'pose' not in arg.resume_posenet:
                    arg.start_epoch = int(stem.split('model_')[-1].split('_')[0]) + 1
                    current_dis = float(stem.split('.pt')[-2].split('_')[-1])
                else:
                    arg.start_epoch = int(stem.split('model_')[-1].split('_')[0]) + 1
                    current_dis = float(stem.split('_pose')[-2].split('_')[-1])

                global_step = (arg.start_epoch - 1) * (len(dataset) * arg.cfg.Train.NUM_EPOCH_REPEAT // cfg.Train.BATCHSIZE)

            print(f'Find Checkpoint {arg.resume_posenet}, Cur_dis:{current_dis:.04f}, Start Epoch {arg.start_epoch}\n')
            estimator = load_part_module(estimator, arg.resume_posenet, False, arg.backbone_oly)
        else:
            estimator = load_part_module(estimator, arg.resume_posenet, False, arg.backbone_oly)

    estimator.cuda()
    criterion.cuda()

    refiner = None
    refin_criterion = None
    optimizer = Optim(estimator.parameters(), lr=cfg.Train.Lr.LR)

    # tensorboard viz
    if not arg.eval_mode and not arg.debug:
        writer = SummaryWriter(arg.log_dir)
    else:
        writer = None

    # lr scheduler init
    lr_scheduler = build_lr_scheduler(arg, optimizer)

    trainer = Trainer(
        estimator, criterion, optimizer, arg, viz=writer, checkpoint=arg.resume_posenet,
        metric=metric, lr_scheduler=lr_scheduler, cfg=arg.cfg,
        refine_model=refiner, refine_criterion=refin_criterion
    )

    if arg.resume_posenet and not refiner:
        trainer.best_test = current_dis
        trainer.global_step = global_step

    if arg.resume_refine_posenet and refiner:
        trainer.best_test = refine_current_dis
        trainer.global_step = global_step


    for epoch in range(arg.start_epoch, arg.nepoch):
        # 高版本Torch修复了这个问题
        # https://github.com/pytorch/pytorch/issues/5059
        np.random.seed(epoch)
        if epoch > arg.cfg.Train.START_POSE_EPOCH or arg.cfg.Train.ENABLE_POSE:
            opt_pose = True
        else:
            opt_pose = False

        if arg.eval_mode:
            arg.mode = 'eval'
            if 'current' in arg.resume_posenet:
                eval_log_name = 'current'
            else:
                eval_log_name = f'{epoch-1}'

            test_logger = setup_logger(
                f'epoch{epoch - 1 if epoch > 0 else 0}_test',
                os.path.join(arg.log_dir, f'epoch_{eval_log_name}_log_{arg.mode}.txt'),
                debug=arg.debug
            )

            trainer.test_epoch(testdataloader, epoch-1, test_logger, 'eval')
            break

        if opt_pose:
            train_logger = setup_logger(f'epoch{epoch}', os.path.join(arg.log_dir, f'epoch_{epoch}_log_pose.txt'), debug=arg.debug)
        else:
            train_logger = setup_logger(f'epoch{epoch}', os.path.join(arg.log_dir, f'epoch_{epoch}_log.txt'),
                                        debug=arg.debug)

        for i in range(arg.cfg.Train.NUM_EPOCH_REPEAT):
            trainer.train_epoch(dataloader, epoch, train_logger)

        if opt_pose:
            test_logger = setup_logger(f'epoch{epoch}_test',
                                       os.path.join(arg.log_dir, f'epoch_{epoch}_test_log_pose.txt'), debug=arg.debug)
        else:
            test_logger = setup_logger(f'epoch{epoch}_test',
                                       os.path.join(arg.log_dir, f'epoch_{epoch}_test_log.txt'), debug=arg.debug)

        trainer.test_epoch(testdataloader, epoch, test_logger, 'test')


if __name__ == '__main__':
    # 出现nan的监测
    # torch.autograd.set_detect_anomaly(True)

    # 打开文件数目增多
    torch.multiprocessing.set_sharing_strategy('file_system')
    train()
