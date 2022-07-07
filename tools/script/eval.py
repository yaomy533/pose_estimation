#**********************************
# 联合测试
# 关键点预测的Pose和xyz用PnP得到的Pose取均值
#**********************************
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('/root/Workspace/project/krrnv2')
from tqdm import tqdm
import torch
import torch.nn
import time
import cv2
import numpy as np
import kornia as kn
import torch.utils.data
from mmcv import Config
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
from dataset.linemod.batchdataset import PoseDataset
from lib.network.krrn import KRRN
from lib.utils.metric import Metric
from lib.utils.utlis import batch_intrinsic_transform
from lib.utils.utlis import load_part_module
# from lib.transform.trans import estimateSimilarityTransform
from lib.transform.umeyama import estimateSimilarityTransform
from tools.trainer import Trainer
from lib.transform.coordinate import crop_resize_by_warp_affine
from lib.transform.allocentric import allo_to_ego_mat_torch
# from lib.network.BPnP import BPnPModle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETCT_PATH = Path(os.path.realpath(__file__)).parent.parent.parent
cfg = Config.fromfile(f'{PROJETCT_PATH}/config/linemod/lm_v3.py')

dataset_root = '/root/Source/ymy_dataset/Linemod_preprocessed'
num_cls = 1
num_kp = 8
cls_type = 'cam'

weight_kps= 0.5
weight_xyz = 0.5

dataset_test = PoseDataset('test', cfg.Data.NUM_POINTS, True, dataset_root, 0.0, 8, cls_type=cls_type, cfg=cfg)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=10)
sym_list = dataset_test.sym_obj
metric = Metric(sym_list)
# BPnP = BPnPModle()
tst_ds_iter = dataloader_test.__iter__()
diameter = dataset_test.diameter[0]

estimator = KRRN(num_cls=1, cfg=cfg)
estimator.cuda()
estimator.eval()
cpkt = '/root/Source/ymy_dataset/trained/krrnv3/linemod/12/pose_model_9_0.010158860717224431_pose.pth'
estimator = load_part_module(estimator, cpkt)
trainer = Trainer(estimator, None, None, None, cfg=cfg)

result = {
    'count':0,

    'xyz_add_dis':0.,
    'xyz_add_suc_10':0.,
    'xyz_add_suc_05':0.,
    'xyz_add_suc_02':0.,


    'reg_add_dis': 0.,
    'reg_add_suc_10': 0.,
    'reg_add_suc_05': 0.,
    'reg_add_suc_02': 0.,

    'fnl_add_dis':0.,
    'fnl_add_suc_10':0.,
    'fnl_add_suc_05':0.,
    'fnl_add_suc_02':0.,
}

pbr = tqdm(enumerate(dataloader_test, 0))

for j, datas in pbr:
    # datas = {k:v.cuda() for k,v in datas.items() if type(v).__name__ == 'Tensor'}
    with torch.no_grad():
        pred = estimator(
            datas['img_croped'].cuda(),
            datas['cloud'].cuda(),
            datas['choose'].cuda(),
            datas['region_point'].cuda(),
            True
        )

    region = pred['region'].cpu()
    xyz_off = pred['xyz'].cpu()
    region = torch.softmax(region, 1)
    region_cls_idx = region.argmax(dim=1, keepdim=True)  # [bs, 1, h, w]
    bs, c, h, w = xyz_off.size()
    n = region.size(1)
    region_point = datas['region_point'].cpu()  # [bs, n, 3]
    # [bs, 1, 3, h, w]
    xyz_base = torch.gather(region_point.view(bs, n, 3, 1, 1).repeat(1, 1, 1, h, w), 1,
                            region_cls_idx.unsqueeze(2).repeat(1, 1, 3, 1, 1))

    xyz = xyz_off + xyz_base.squeeze(1)

    bs = xyz.size(0)

    roi_extents = datas['extent']
    left_border = datas['lfborder']
    _choose = datas['choose']
    num_points = 256
    choose_idx = torch.randperm(_choose.size(-1), device=_choose.device)[:num_points]
    choose = _choose[..., choose_idx]  # [bs, 1, 500]

    x_map_choosed = datas['x_map_choosed'][:, choose_idx]
    y_map_choosed = datas['y_map_choosed'][:, choose_idx]
    _intrinsic = datas['intrinsic']
    pts2d_res = torch.cat([x_map_choosed, y_map_choosed], -1)
    K_matrix = batch_intrinsic_transform(_intrinsic)
    coordintae = torch.cat([
        xyz[:, 0:1, :, :] * roi_extents[:, 0:1].view(-1, 1, 1, 1) + left_border[:, 0:1].view(-1, 1, 1, 1),
        xyz[:, 1:2, :, :] * roi_extents[:, 1:2].view(-1, 1, 1, 1) + left_border[:, 1:2].view(-1, 1, 1, 1),
        xyz[:, 2:3, :, :] * roi_extents[:, 2:3].view(-1, 1, 1, 1) + left_border[:, 2:3].view(-1, 1, 1, 1)
    ], dim=1).permute(0, 2, 3, 1)

    coordinate_choosed = torch.gather(coordintae.view(bs, -1, 3), -2, choose.permute(0, 2, 1).repeat(1, 1, 3))
    coordinate_choosed_all = torch.gather(coordintae.view(bs, -1, 3), -2, _choose.permute(0, 2, 1).repeat(1, 1, 3))


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
    xyz_r = out_rx
    xyz_t = out_tx

    cloud = datas['cloud']
    model_points = datas['model_points']
    target = datas['target'][0]
    # s, dpr, dpt, _ = estimateSimilarityTransform(coordinate_choosed_all[0].numpy(), cloud[0].numpy())
    #
    # try:
    #     dpr, dpt = torch.from_numpy(dpr.astype(np.float32)).unsqueeze(0), torch.from_numpy(dpt.astype(np.float32)).unsqueeze(0).unsqueeze(1)
    # except AttributeError:
    #     continue

    reg_r, reg_t = pred['pred_r'].cpu(), pred['pred_t'].cpu()
    fnl_r, fnl_t = xyz_r, reg_t

    # final_r = base_r @ res_r
    # final_t = base_t + res_t.unsqueeze(1) @ base_r.permute(0, 2, 1)

    target_r = datas['target_r'].clone()
    target_t = datas['target_t'].clone()

    dis_xyz_r = metric.angular_distance(xyz_r, target_r)
    dis_reg_r = metric.angular_distance(reg_r, target_r)
    dis_fnl_r = metric.angular_distance(fnl_r, target_r)

    dis_xyz_t = metric.translation_distance(xyz_t, target_t)
    dis_reg_t = metric.translation_distance(reg_t, target_t)
    dis_fnl_t = metric.translation_distance(fnl_t, target_t)

    pred_xyz_point = model_points @ xyz_r.permute(0, 2, 1) + xyz_t
    pred_reg_point = model_points @ reg_r.permute(0, 2, 1) + reg_t
    pred_fnl_point = model_points @ fnl_r.permute(0, 2, 1) + fnl_t

    dis_xyz_add, _ = metric.cal_adds_cuda(pred_xyz_point[0], target, 0)
    dis_reg_add, _ = metric.cal_adds_cuda(pred_reg_point[0], target, 0)
    dis_fnl_add, _ = metric.cal_adds_cuda(pred_fnl_point[0], target, 0)

    # xyz_r_qua = kn.geometry.rotation_matrix_to_quaternion(xyz_r)
    # kps_r_qua = kn.geometry.rotation_matrix_to_quaternion(base_r)
    # final_r_qua = xyz_r_qua*weight_xyz + kps_r_qua*weight_kps
    # final_r = kn.geometry.quaternion_to_rotation_matrix(final_r_qua)
    # final_t = base_t*weight_kps + xyz_t*weight_xyz
    #
    # dis_final_t = metric.translation_distance(final_t, target_t)
    # dis_final_r = metric.angular_distance(final_r, target_r)
    # pred_final_point = model_points @ final_r.permute(0, 2, 1) + final_t
    # dis_final_add, _ = metric.cal_adds_cuda(pred_final_point[0], target, 0)
    pbr.set_description(
        f"fnl_add:{dis_fnl_add:.04f}, fnl_r:{dis_fnl_r.item():.04e}, fnl_t:{dis_fnl_t.item():.04f}, "
        f"reg_add:{dis_reg_add:.04f}, regf_r:{dis_reg_r.item():.04e}, reg_t:{dis_reg_t.item():.04f}, "
        f"xyz_add:{dis_xyz_add:.04f}, xyz_r:{dis_xyz_r.item():.04e}, xyz_t:{dis_xyz_t.item():.04f}, "
    )

    result['count'] += 1
    
    result['xyz_add_dis'] += dis_xyz_add
    if dis_xyz_add < 0.1*diameter:
        result['xyz_add_suc_10'] += 1
        if dis_xyz_add < 0.05*diameter:
            result['xyz_add_suc_05'] += 1
            if dis_xyz_add < 0.02*diameter:
                result['xyz_add_suc_02'] += 1

    result['reg_add_dis'] += dis_reg_add
    if dis_reg_add < 0.1*diameter:
        result['reg_add_suc_10'] += 1
        if dis_reg_add < 0.05*diameter:
            result['reg_add_suc_05'] += 1
            if dis_reg_add < 0.02*diameter:
                result['reg_add_suc_02'] += 1


    result['fnl_add_dis'] += dis_fnl_add
    if dis_fnl_add < 0.1*diameter:
        result['fnl_add_suc_10'] += 1
        if dis_fnl_add < 0.05*diameter:
            result['fnl_add_suc_05'] += 1
            if dis_fnl_add < 0.02*diameter:
                result['fnl_add_suc_02'] += 1


print(f"xyz_add_dis: {float(result['xyz_add_dis']) / float(result['count']) :.04f}",  
        f"xyz_add_suc_10: {float(result['xyz_add_suc_10']) / float(result['count']) :.04f} ",
        f"xyz_add_suc_05: {float(result['xyz_add_suc_05']) / float(result['count']):.04f} ",
        f"xyz_add_suc_10: {float(result['xyz_add_suc_02']) / float(result['count']):.04f} "
    )

print(f"reg_add_dis: {float(result['reg_add_dis']) / float(result['count']) :.04f}",
        f"reg_add_suc_10: {float(result['reg_add_suc_10']) / float(result['count']) :.04f} ",
        f"reg_add_suc_05: {float(result['reg_add_suc_05']) / float(result['count']):.04f} ",
        f"reg_add_suc_10: {float(result['reg_add_suc_02']) / float(result['count']):.04f} "
    )

print(f"fnl_add_dis: {float(result['fnl_add_dis']) / float(result['count']) :.04f}",  
        f"fnl_add_suc_10: {float(result['fnl_add_suc_10']) / float(result['count']) :.04f} ",
        f"fnl_add_suc_05: {float(result['fnl_add_suc_05']) / float(result['count']):.04f} ",
        f"fnl_add_suc_10: {float(result['fnl_add_suc_02']) / float(result['count']):.04f} "
    )
