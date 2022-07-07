#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 19:50
# @Author  : yaomy

import random
import torch
import time
import copy
import kornia as kn
import numpy as np
import torch.nn.functional as F
import cv2
from torch.cuda.amp import autocast
from torch.autograd import Variable
from lib.network.dnn.BPnP import BPnPModle
from lib.network.krrn import KRRN
from lib.utils.utlis import batch_intrinsic_transform, choose_ransac_batch
from lib.transform.coordinate import ortho6d_to_mat_batch
from lib.transform.allocentric import allo_to_ego_mat_torch


class Trainer:
    def __init__(self,
                 model: KRRN,
                 criterion,
                 optimizer,
                 opt,
                 viz=None,
                 checkpoint=None,
                 metric=None,
                 lr_scheduler=None,
                 cfg=None,
                 refine_model=None,
                 refine_criterion=None,
                 ):
        super(Trainer, self).__init__()
        self.model, self.criterion, self.opt, self.checkpoint, self.optimizer, self.viz, self.metric, self.lr_scheduler = (
            model,
            criterion,
            opt,
            checkpoint,
            optimizer,
            viz,
            metric,
            lr_scheduler,
        )
        self.cfg = cfg
        self.best_test = np.Inf
        self.global_step = 0
        self.st_time = time.time()
        self.BPnP = BPnPModle()
        self.BPnP.cuda()
        self.viz_list = [100, 200, 300, 400]
        self.objlist = []
        self.refine_model = refine_model
        self.refine_criterion = refine_criterion
        if self.refine_model is not None:
            self.refine = True
        else:
            self.refine = False

        self.mul_scale_datas = []
        self.mul_scales = []
        self.mul_scales_count = []
        if self.refine:
            self.bs = self.cfg.Train.RF_BATCHSIZE
        else:
            self.bs = self.cfg.Train.BATCHSIZE

    def train_epoch(self, train_dataloader, epoch, logger):
        batch_count = 0
        self.model.train()
        self.optimizer.zero_grad()
        if epoch > self.cfg.Train.START_POSE_EPOCH or self.cfg.Train.ENABLE_POSE:
            opt_pose = True
        else:
            opt_pose = False

        for i, patch_datas in enumerate(train_dataloader, 0):
            datas = self.process_patch_datas(patch_datas)
            if datas is None:
                continue
            img_size = datas['img_croped'].size(2)
            if img_size > 256:
                continue

            self.global_step += 1
            if batch_count >= self.opt.epoch_step:
                break

            pred, loss_dict = self.forward(datas, grad=True, opt_pose=opt_pose)
            # try:
            #     pred, loss_dict = self.forward(datas, grad=True, opt_pose=opt_pose)
            # except RuntimeError:
            #     print(f'out of memery, size: {img_size}')
            #     continue

            batch_count += 1
            if not torch.any(torch.isnan(loss_dict['loss'])):
                # self.optimizer.zero_grad()
                loss_dict['loss'].backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                torch.save(self.model.state_dict(), '{0}/pose_model_nan_{1}.pth'.format(self.opt.outf, batch_count))
                self.optimizer.zero_grad()

            content = f'Train time {self._time} ' + f'Epoch {epoch} Batch {batch_count} Frame {batch_count * self.opt.batchsize} '

            for k, v in loss_dict.items():
                if isinstance(v, float) or isinstance(v, int):
                    if 'loss' in k:
                        content += f'{k}:{v: .04f} '
                        if self.viz:
                            self.viz.add_scalar(k, v, self.global_step)
                else:
                    if 'loss' in k:
                        content += f'{k}:{v.detach().cpu().numpy().item(): .04f} '
                        if self.viz:
                            self.viz.add_scalar(k, v.detach().cpu().numpy().item(), self.global_step)

            if self.lr_scheduler is not None and self.cfg.Train.Lr.LR_SCHEDULER in ['step', 'lambda']:
                self.lr_scheduler.step()

            cur_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if self.viz:
                self.viz.add_scalar('lr', cur_lr, self.global_step)
            content += f'lr:{cur_lr:.4e} '
            content += f'img: {img_size:03d} '
            logger.info(content)

            # delete caches, release the memory of cuda
            # del datas, loss_dict
            # torch.cuda.empty_cache()

            if batch_count != 0 and batch_count % 2000 == 0:
                torch.save(self.model.state_dict(), '{0}/pose_model_current.pth'.format(self.opt.outf))

        if self.lr_scheduler is not None and self.cfg.Train.Lr.LR_SCHEDULER == 'epoch':
            self.lr_scheduler.step()

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

    def test_epoch(self, test_dataloader, test_epoch, test_logger, mode='test'):
        self.model.eval()
        test_logger.info(f'{mode} time {self._time}, Testing started')
        self.objlist = test_dataloader.dataset.objlist
        result = copy.copy(self.result)

        if test_epoch > self.cfg.Train.START_POSE_EPOCH or self.cfg.Train.ENABLE_POSE:
            opt_pose = True
        else:
            opt_pose = False

        rotation_dis_threshold = 5.0
        translation_dis_threshold = 0.05
        test_count = 0
        test_dis = 0
        xyz_dis = 0
        normal_dis = 0
        mask_dis = 0
        base_succ = 0
        refine_succ = 0

        for j, datas in enumerate(test_dataloader, 0):
            pred, loss_dict = self.forward(datas, grad=False, opt_pose=opt_pose)
            cls_id = datas['cls_id'].cuda()
            obj_id = self.objlist[cls_id]
            base_r, base_t = self.get_pose(pred, datas)
            base_r : torch.tensor
            base_add_dis, base_r_dis, base_t_dis = self.cal_dis(base_r, base_t, datas)

            diameter = self.opt.diameter[cls_id.item()]
            test_count += 1
            result['all_num'][obj_id] += 1
            result['obj_num'][obj_id] += 1

            result['dis_base_rt'][obj_id] += base_add_dis
            result['dis_base_r'][obj_id] += base_r_dis
            result['dis_base_t'][obj_id] += base_t_dis
            result['dis_xyz'][obj_id] += loss_dict['loss_xyz'].detach().cpu().item()
            result['dis_mask'][obj_id] += loss_dict['loss_mask'].detach().cpu().item()
            result['dis_normal'][obj_id] += loss_dict['loss_normal'].detach().cpu().item()

            if base_add_dis < 0.1 * diameter:
                result['succ_base_rt'][obj_id] += 1
            if base_r_dis < rotation_dis_threshold:
                result['succ_base_r'][obj_id] += 1
            if base_t_dis < translation_dis_threshold:
                result['succ_base_t'][obj_id] += 1

            xyz_dis += loss_dict['loss_xyz'].detach().cpu().item()
            normal_dis += loss_dict['loss_normal'].detach().cpu().item()
            mask_dis += loss_dict['loss_mask'].detach().cpu().item()

            if opt_pose:
                reg_r, reg_t = base_r.cuda(), pred['pred_t'].cuda()
                final_r, final_t = base_r.cuda(), reg_t.cuda()
                reg_add_dis, reg_r_dis, reg_t_dis = self.cal_dis(reg_r, reg_t, datas)
                final_add_dis, final_r_dis, final_t_dis = self.cal_dis(final_r, final_t, datas)

                result['dis_reg_rt'][obj_id] += reg_add_dis
                result['dis_reg_r'][obj_id] += reg_r_dis
                result['dis_reg_t'][obj_id] += reg_t_dis

                result['dis_final_rt'][obj_id] += final_add_dis
                result['dis_final_r'][obj_id] += final_r_dis
                result['dis_final_t'][obj_id] += final_t_dis

                if reg_add_dis < 0.1 * diameter:
                    result['succ_reg_rt'][obj_id] += 1
                if reg_r_dis < rotation_dis_threshold:
                    result['succ_reg_r'][obj_id] += 1
                if reg_t_dis < translation_dis_threshold:
                    result['succ_reg_t'][obj_id] += 1

                test_dis += final_add_dis
                if final_add_dis < 0.1 * diameter:
                    refine_succ += 1
                    result['succ_final_rt'][obj_id] += 1
                    content = f'Obj {obj_id:02d} Yep Pass! {mode} ' + f'Epoch {test_epoch} Batch {test_count:05d} ' \
                        + f'final_dis:{final_add_dis:.04f} r:{final_r_dis:5.2f} t:{final_t_dis:.04f} ' \
                        + f'reg_dis:{reg_add_dis:.04f} r:{reg_r_dis:5.2f} t:{reg_t_dis:.04f} ' \
                        + f'base_dis:{base_add_dis:.04f} r:{base_r_dis:5.2f} t:{base_t_dis:.04f} ' \
                        + f'loss_xyz:{loss_dict["loss_xyz"].detach().cpu().item():.04f} ' \
                        + f'loss_region:{loss_dict["loss_region"].detach().cpu().item():.04f} ' \
                        + f'loss_normal:{loss_dict["loss_normal"].detach().cpu().item():.04f} ' \
                        + f'loss_mask:{loss_dict["loss_mask"].detach().cpu().item():.04f} '
                else:
                    content = f'Obj {obj_id:02d} Not Pass! {mode} ' + f'Epoch {test_epoch} Batch {test_count:05d} ' \
                        + f'final_dis:{final_add_dis:.04f} r:{final_r_dis:5.2f} t:{final_t_dis:.04f} ' \
                        + f'reg_dis:{reg_add_dis:.04f} r:{reg_r_dis:5.2f} t:{reg_t_dis:.04f} ' \
                        + f'base_dis:{base_add_dis:.04f} r:{base_r_dis:5.2f} t:{base_t_dis:.04f} ' \
                        + f'loss_xyz:{loss_dict["loss_xyz"].detach().cpu().item():.04f} ' \
                        + f'loss_region:{loss_dict["loss_region"].detach().cpu().item():.04f} ' \
                        + f'loss_normal:{loss_dict["loss_normal"].detach().cpu().item():.04f} ' \
                        + f'loss_mask:{loss_dict["loss_mask"].detach().cpu().item():.04f} '

                if final_r_dis < rotation_dis_threshold:
                    result['succ_final_r'][obj_id] += 1
                if final_t_dis < translation_dis_threshold:
                    result['succ_final_t'][obj_id] += 1
                if base_add_dis < 0.1 * diameter:
                    base_succ += 1

            else:
                test_dis += base_add_dis
                if base_add_dis < 0.1 * diameter:
                    base_succ += 1
                    content = f'Obj {obj_id:02d} Yep Pass! {mode} ' + f'Epoch {test_epoch} Batch {test_count:05d} ' \
                        + f'base_dis:{base_add_dis:.04f} r:{base_r_dis:5.2f} t:{base_t_dis:.04f} ' \
                        + f'loss_xyz:{loss_dict["loss_xyz"].detach().cpu().item():.04f} ' \
                        + f'loss_normal:{loss_dict["loss_normal"].detach().cpu().item():.04f} ' \
                        + f'loss_mask:{loss_dict["loss_mask"].detach().cpu().item():.04f} '
                else:
                    content = f'Obj {obj_id:02d} Not Pass! {mode} ' + f'Epoch {test_epoch} Batch {test_count:05d} ' \
                        + f'base_dis:{base_add_dis:.04g} r:{base_r_dis:5.2f} t:{base_t_dis:.04f} ' \
                        + f'loss_xyz:{loss_dict["loss_xyz"].detach().cpu().item():.04f} ' \
                        + f'loss_normal:{loss_dict["loss_normal"].detach().cpu().item():.04f} ' \
                        + f'loss_mask:{loss_dict["loss_mask"].detach().cpu().item():.04f} '

            test_logger.info(content)


        test_dis = test_dis / test_count
        xyz_dis = xyz_dis / test_count
        normal_dis = normal_dis / test_count
        mask_dis = mask_dis / test_count
        base_succ = base_succ / test_count
        refine_succ = refine_succ / test_count

        if self.viz:
            self.viz.add_scalar('test_dis_add', test_dis, test_epoch)
            self.viz.add_scalar('test_dis_xyz', xyz_dis, test_epoch)
            self.viz.add_scalar('test_dis_normal', normal_dis, test_epoch)
            self.viz.add_scalar('test_dis_mask', mask_dis, test_epoch)
            self.viz.add_scalar('test_base_succ', base_succ, test_epoch)
            self.viz.add_scalar('test_refine_succ', refine_succ, test_epoch)

        for i in self.objlist:
            if opt_pose:
                content = f'Object {i} Fnl Dis:{float(result["dis_final_rt"][i]) / float(result["obj_num"][i]):.04f} Succ:{float(result["succ_final_rt"][i]) / float(result["obj_num"][i]):.03f}\n' \
                          + f'Base ADD-S Dis:{float(result["dis_base_rt"][i]) / float(result["obj_num"][i]):.04f} Succ:{float(result["succ_base_rt"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'5° Dis:{float(result["dis_base_r"][i]) / float(result["obj_num"][i]):5.2f} Succ:{float(result["succ_base_r"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'5CM Dis:{float(result["dis_base_t"][i]) / float(result["obj_num"][i]):.04f} Succ:{float(result["succ_base_t"][i]) / float(result["obj_num"][i]):.03f};' \
                          + f'\n' \
                          + f'Regs ADD-S Dis:{float(result["dis_reg_rt"][i]) / float(result["obj_num"][i]):.04f} Succ:{float(result["succ_reg_rt"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'5° Dis:{float(result["dis_reg_r"][i]) / float(result["obj_num"][i]):5.2f} Succ:{float(result["succ_reg_r"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'5CM Dis:{float(result["dis_reg_t"][i]) / float(result["obj_num"][i]):.04f} Succ:{float(result["succ_reg_t"][i]) / float(result["obj_num"][i]):.03f};' \
                          + f'\n' \
                          + f'Fnal ADD-S Dis:{float(result["dis_final_rt"][i]) / float(result["obj_num"][i]):.04f} Succ:{float(result["succ_final_rt"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'5° Dis:{float(result["dis_final_r"][i]) / float(result["obj_num"][i]):5.2f} Succ:{float(result["succ_final_r"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'5CM Dis:{float(result["dis_final_t"][i]) / float(result["obj_num"][i]):.04f} Succ:{float(result["succ_final_t"][i]) / float(result["obj_num"][i]):.03f};' \
                          + f'\n' \
                          + f'XYZ Dis: {float(result["dis_xyz"][i]) / float(result["obj_num"][i]):.04f}; ' \
                          + f'NML Dis: {float(result["dis_normal"][i]) / float(result["obj_num"][i]):.04f}; ' \
                          + f'MASK Dis: {float(result["dis_mask"][i]) / float(result["obj_num"][i]):.04f}; '
            else:
                content = f'Object {i} ' \
                          + f'Base ADD-S Dis:{float(result["dis_base_rt"][i]) / float(result["obj_num"][i]):.04f} Succ:{float(result["succ_base_rt"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'5° Dis:{float(result["dis_base_r"][i]) / float(result["obj_num"][i]):5.2f} Succ:{float(result["succ_base_r"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'5CM Dis:{float(result["dis_base_t"][i]) / float(result["obj_num"][i]):.04g} Succ:{float(result["succ_base_t"][i]) / float(result["obj_num"][i]):.03f}; ' \
                          + f'XYZ Dis: {float(result["dis_xyz"][i]) / float(result["obj_num"][i]):.04f}; ' \
                          + f'NML Dis: {float(result["dis_normal"][i]) / float(result["obj_num"][i]):.04f}; ' \
                          + f'MASK Dis: {float(result["dis_mask"][i]) / float(result["obj_num"][i]):.04f}; '

            test_logger.info(content)
        def get_sum(d:dict):
            return float(sum(d.values()))

        test_logger.info('\n')
        all_content = f'\n' \
                          + f'Base ADD-S Dis:{get_sum(result["dis_base_rt"]) / get_sum(result["obj_num"]):.04f} Succ:{get_sum(result["succ_base_rt"]) / get_sum(result["obj_num"]):.03f}; ' \
                          + f'5° Dis:{get_sum(result["dis_base_r"]) / get_sum(result["obj_num"]):5.2f} Succ:{get_sum(result["succ_base_r"]) / get_sum(result["obj_num"]):.03f}; ' \
                          + f'5CM Dis:{get_sum(result["dis_base_t"]) / get_sum(result["obj_num"]):.04f} Succ:{get_sum(result["succ_base_t"]) / get_sum(result["obj_num"]):.03f};' \
                          + f'\n' \
                          + f'Regs ADD-S Dis:{get_sum(result["dis_reg_rt"]) / get_sum(result["obj_num"]):.04f} Succ:{get_sum(result["succ_reg_rt"]) / get_sum(result["obj_num"]):.03f}; ' \
                          + f'5° Dis:{get_sum(result["dis_reg_r"]) / get_sum(result["obj_num"]):5.2f} Succ:{get_sum(result["succ_reg_r"]) / get_sum(result["obj_num"]):.03f}; ' \
                          + f'5CM Dis:{get_sum(result["dis_reg_t"]) / get_sum(result["obj_num"]):.04f} Succ:{get_sum(result["succ_reg_t"]) / get_sum(result["obj_num"]):.03f};' \
                          + f'\n' \
                          + f'Fnal ADD-S Dis:{get_sum(result["dis_final_rt"]) / get_sum(result["obj_num"]):.04f} Succ:{get_sum(result["succ_final_rt"]) / get_sum(result["obj_num"]):.03f}; ' \
                          + f'5° Dis:{get_sum(result["dis_final_r"]) / get_sum(result["obj_num"]):5.2f} Succ:{get_sum(result["succ_final_r"]) / get_sum(result["obj_num"]):.03f}; ' \
                          + f'5CM Dis:{get_sum(result["dis_final_t"]) / get_sum(result["obj_num"]):.04f} Succ:{get_sum(result["succ_final_t"]) / get_sum(result["obj_num"]):.03f};' \
                          + f'\n' \
                          + f'XYZ Dis: {get_sum(result["dis_xyz"]) / get_sum(result["obj_num"]):.04f}; ' \
                          + f'NML Dis: {get_sum(result["dis_normal"]) / get_sum(result["obj_num"]):.04f}; ' \
                          + f'MASK Dis: {get_sum(result["dis_mask"]) / get_sum(result["obj_num"]):.04f}; '

        test_logger.info(all_content)

        # if not opt_pose:
        #     summary_keys = ["dis_base_rt", "succ_base_rt", "dis_base_r", "succ_base_r", "dis_base_t", "succ_base_t",  "dis_kps", "dis_xyz"]
        #     summary = {k: self.sum_dict(result[k]) / self.sum_dict(result["obj_num"]) for k in summary_keys}
        #     content = f'All Object ' \
        #               + f'Base ADD-S Dis:{float(summary["dis_base_rt"]):.04f} Succ:{float(summary["succ_base_rt"]):.03f}; ' \
        #               + f'5° Dis:{float(summary["dis_base_r"]):.04f} Succ:{float(summary["succ_base_r"]):.03f}; ' \
        #               + f'5CM Dis:{float(summary["dis_base_t"]):.04f} Succ:{float(summary["succ_base_t"]):.03f}; ' \
        #               + f'Kps Dis: {float(summary["dis_kps"]):.04f}; ' \
        #               + f'XYZ Dis: {float(summary["dis_xyz"]):.04f}; '
        #     test_logger.info(content)
        # else:
        #     summary_keys = ["dis_base_rt", "succ_base_rt", "dis_base_r", "succ_base_r", "dis_base_t", "succ_base_t",  "dis_kps", "dis_xyz"]
        #     summary = {k: self.sum_dict(result[k]) / self.sum_dict(result["obj_num"]) for k in summary_keys}
        #     content = f'All Object ' \
        #               + f'Base ADD-S Dis:{float(summary["dis_base_rt"]):.04f} Succ:{float(summary["succ_base_rt"]):.03f}; ' \
        #               + f'5° Dis:{float(summary["dis_base_r"]):.04f} Succ:{float(summary["succ_base_r"]):.03f}; ' \
        #               + f'5CM Dis:{float(summary["dis_base_t"]):.04f} Succ:{float(summary["succ_base_t"]):.03f}; ' \
        #               + f'Kps Dis: {float(summary["dis_kps"]):.04f}; ' \
        #               + f'XYZ Dis: {float(summary["dis_xyz"]):.04f}; '
        #     test_logger.info(content)
        print('>>>>>>>>----------epoch {0} {1} finish---------<<<<<<<<'.format(test_epoch, mode))

        if mode == 'test':
            if test_dis <= self.best_test:
                self.best_test = test_dis
                if not opt_pose:
                    torch.save(self.model.state_dict(),
                               '{0}/pose_model_{1}_{2}.pth'.format(self.opt.outf, test_epoch, test_dis))
                else:
                    torch.save(self.model.state_dict(),
                               '{0}/pose_model_{1}_{2}_pose.pth'.format(self.opt.outf, test_epoch, test_dis))
                print(f'>>>>>>>>----------EPOCH {test_epoch} BEST TEST MODEL SAVED---------<<<<<<<<')

            if self.cfg.Train.Lr.LR_SCHEDULER == 'manual' and not self.opt.decay_start and self.best_test < self.cfg.Train.Lr.Manual.DECAY_MARGIN:
                self.opt.decay_start = True
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.cfg.Train.Lr.Manual.DECAY_RATE

    def cal_dis(self, pred_r, pred_t, datas):
        device = pred_t.device
        target_r = datas['target_r'].to(device)
        target_t = datas['target_t'].to(device)
        target = datas['target'].to(device)
        cls_id = datas['cls_id'].to(device)
        pred_points = datas['model_points'].to(device) @ pred_r.permute(0, 2, 1) + pred_t
        add_dis, _ = self.metric.cal_adds_cuda(pred_points[0], target[0], cls_id.item())

        r_dis = self.metric.angular_distance(pred_r, target_r).squeeze().item()
        t_dis = self.metric.translation_distance(pred_t.squeeze(dim=1), target_t).squeeze().item()
        return add_dis, r_dis, t_dis

    @staticmethod
    def get_pose(pred, data):
        # xyz = pred['xyz'].cpu()
        region = pred['region'].cpu()
        xyz_off = pred['xyz'].cpu()

        # region = torch.softmax(region, 1)

        # region_cls_idx = region.argmax(dim=1, keepdim=True)   # [bs, 1, h, w]
        # bs, c, h, w = xyz_off.size()
        # n = region.size(1)
        # region_point = data['region_point'].cpu() # [bs, n, 3]
        # # [bs, 1, 3, h, w]
        # xyz_base = torch.gather(region_point.view(bs, n, 3, 1, 1).repeat(1, 1, 1, h, w), 1, region_cls_idx.unsqueeze(2).repeat(1, 1, 3, 1, 1))
        # xyz = xyz_off + xyz_base

        xyz = xyz_off

        bs = xyz.size(0)
        assert bs == 1
        roi_extents = data['extent']
        left_border = data['lfborder']
        _choose = data['choose']
        num_points = 256
        choose_idx = torch.randperm(_choose.size(-1), device=_choose.device)[:num_points]
        choose = _choose[..., choose_idx]  # [bs, 1, 500]

        x_map_choosed = data['x_map_choosed'][:, choose_idx]
        y_map_choosed = data['y_map_choosed'][:, choose_idx]
        _intrinsic = data['intrinsic']
        pts2d_res = torch.cat([x_map_choosed, y_map_choosed], -1)
        K_matrix = batch_intrinsic_transform(_intrinsic)
        coordintae = torch.cat([
            xyz[:, 0:1, :, :] * roi_extents[:, 0:1].view(-1, 1, 1, 1) + left_border[:, 0:1].view(-1, 1, 1, 1),
            xyz[:, 1:2, :, :] * roi_extents[:, 1:2].view(-1, 1, 1, 1) + left_border[:, 1:2].view(-1, 1, 1, 1),
            xyz[:, 2:3, :, :] * roi_extents[:, 2:3].view(-1, 1, 1, 1) + left_border[:, 2:3].view(-1, 1, 1, 1)
        ], dim=1).permute(0, 2, 3, 1)

        coordinate_choosed = torch.gather(coordintae.view(bs, -1, 3), -2, choose.permute(0, 2, 1).repeat(1, 1, 3))

        _, rvec, T, inliners = cv2.solvePnPRansac(
            objectPoints=coordinate_choosed[0].numpy(), imagePoints=pts2d_res[0].numpy(),
            cameraMatrix=K_matrix[0].numpy(),
            distCoeffs=None, flags=cv2.SOLVEPNP_EPNP, confidence=0.9999, reprojectionError=1
        )

        angle_axis = torch.tensor(rvec, dtype=torch.float).view(bs, 3)
        T = torch.tensor(T, dtype=torch.float).view(bs, 3)
        P_out = torch.cat((angle_axis, T), dim=-1)

        out_rx = kn.geometry.conversions.angle_axis_to_rotation_matrix(
            P_out[:, 0:3].contiguous().view(bs, 3)
        )
        out_tx = P_out[:, 3:6]

        return out_rx, out_tx

    def forward(self, datas, grad=True, opt_pose=False):
        # datas = {k:v.cuda() for k,v in datas.items() if type(v).__name__ == 'Tensor'}
        gt = {
            'xyz': datas['xyz'].cuda(),
            'normal': datas['normal'].cuda(),
            'region': datas['region'].cuda(),
            'mask': datas['mask'].cuda(),
            'multi_cls_mask': datas['multi_cls_mask'].cuda(),
            'target': datas['target'].cuda(),
            'model_points': datas['model_points'].cuda(),
            'cls_id': datas['cls_id'].cuda(),
            'target_r': datas['target_r'].cuda(),
            'region_point': datas['region_point'].cuda()
        }

        if grad:
            with autocast(enabled=self.cfg.Train.AMP):
                pred = self.model(
                    Variable(datas['img_croped']).cuda(),
                    Variable(datas['cloud']).cuda(),
                    Variable(datas['choose']).cuda(),
                    Variable(datas['cls_id']).cuda(),
                    Variable(datas['xyz']).cuda(),
                    opt_pose
                )
                loss_dict = self.criterion(pred, gt, opt_pose=opt_pose)
        else:
            with torch.no_grad():
                pred = self.model(
                    datas['img_croped'].cuda(),
                    datas['cloud'].cuda(),
                    datas['choose'].cuda(),
                    datas['cls_id'].cuda(),
                    datas['xyz'].cuda(),
                    opt_pose
                )
                loss_dict = self.criterion(pred, gt, opt_pose=opt_pose)
        return pred, loss_dict

    @property
    def result(self):
        return {
            'all_num': {i: 0. for i in self.objlist},
            'obj_num': {i: 0. for i in self.objlist},  # 物体数目统计

            'succ_base_rt': {i: 0. for i in self.objlist},
            'dis_base_rt': {i: 0. for i in self.objlist},
            'succ_base_r': {i: 0. for i in self.objlist},
            'dis_base_r': {i: 0. for i in self.objlist},
            'succ_base_t': {i: 0. for i in self.objlist},
            'dis_base_t': {i: 0. for i in self.objlist},

            'succ_reg_rt': {i: 0. for i in self.objlist},
            'dis_reg_rt': {i: 0. for i in self.objlist},
            'succ_reg_r': {i: 0. for i in self.objlist},
            'dis_reg_r': {i: 0. for i in self.objlist},
            'succ_reg_t': {i: 0. for i in self.objlist},
            'dis_reg_t': {i: 0. for i in self.objlist},

            'succ_final_rt': {i: 0. for i in self.objlist},
            'dis_final_rt': {i: 0. for i in self.objlist},
            'succ_final_r': {i: 0. for i in self.objlist},
            'dis_final_r': {i: 0. for i in self.objlist},
            'succ_final_t': {i: 0. for i in self.objlist},
            'dis_final_t': {i: 0. for i in self.objlist},

            'dis_mask': {i: 0. for i in self.objlist},
            'dis_normal': {i: 0. for i in self.objlist},
            'dis_xyz': {i: 0. for i in self.objlist},
        }

    @property
    def _time(self):
        """ The time since training or testing began
        """
        return time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - self.st_time))

    @staticmethod
    def sum_dict(x):
        return sum([v for k, v in x.items()])

    def process_patch_datas(self, patch_datas):
        # st = time.time()
        for datas in patch_datas:
            comb_c, _, bw, _ = datas['img_croped'].size()
            if bw in self.mul_scales:
                idx = self.mul_scales.index(bw)
                self.mul_scales_count[idx] += comb_c
                if self.mul_scale_datas[idx] == dict():
                    self.mul_scale_datas[idx] = datas
                # mul_scale_datas[idx] = default_collate([mul_scale_datas[idx], datas])
                self.mul_scale_datas[idx] = {k: torch.cat([v, datas[k]], dim=0) for k, v in self.mul_scale_datas[idx].items() if
                                        isinstance(v, torch.Tensor)}
            else:
                self.mul_scale_datas.append(datas)
                self.mul_scales.append(bw)
                self.mul_scales_count.append(comb_c)

        # print('--->', time.time()-st, self.mul_scale_datas.__len__(), self.mul_scales, self.mul_scales_count)
        items = np.where(np.asarray(self.mul_scales_count) > self.bs)[0].tolist()
        if len(items) == 0:
            return None

        item = random.choice(items)
        # 取出[0:bs]
        tmp = {k: v[:self.bs] for k, v in self.mul_scale_datas[item].items()}

        # 删除[0:bs]
        self.mul_scale_datas[item] = {k: v[self.bs:] for k, v in self.mul_scale_datas[item].items() if isinstance(v, torch.Tensor)}
        self.mul_scales_count[item] -= self.bs
        # print('--->', time.time()-st, self.mul_scale_datas.__len__(), self.mul_scales, self.mul_scales_count)
        return tmp


