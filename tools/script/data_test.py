#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 16:51
# @Author  : yaomy
import random
import torch
from dataset.linemod.batchdataset import PoseDataset
import numpy as np

import re
import collections
from torch._six import string_classes
np_str_obj_array_pattern = re.compile(r'[SaUO]')

def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        # array of string classes and object
        if elem_type.__name__ == 'ndarray' \
                and np_str_obj_array_pattern.search(data.dtype.str) is not None:
            return data
        return torch.as_tensor(data)
    elif isinstance(data, collections.abc.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return [default_convert(d) for d in data]
    else:
        return data


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def my_colla_fn(batch):
    ws = []
    new_batch_groups = list()
    for b in batch:
        bw = b['img_croped'].size(1)
        if bw not in ws:
            ws.append(bw)
            new_batch_groups.append([])
            new_batch_groups[-1].append(b)
        else:
            new_batch_groups[ws.index(bw)].append(b)
    result = []
    for new_bactch in new_batch_groups:
        result.append(default_collate(new_bactch))
    return result

def process_patch_datas(patch_datas, bs):
    # combine data
    for datas in patch_datas:
        comb_c, _, bw, _ = datas['img_croped'].size()
        if bw in mul_scales:
            idx = mul_scales.index(bw)
            mul_scales_count[idx] += comb_c
            if mul_scale_datas[idx] == dict():
                mul_scale_datas[idx] = datas
            # mul_scale_datas[idx] = default_collate([mul_scale_datas[idx], datas])
            mul_scale_datas[idx] = {k: torch.cat([v, datas[k]], dim=0) for k, v in mul_scale_datas[idx].items() if isinstance(v, torch.Tensor)}
        else:
            mul_scale_datas.append(datas)
            mul_scales.append(bw)
            mul_scales_count.append(comb_c)

    items = np.where(np.asarray(mul_scales_count) > bs)[0].tolist()
    if len(items) == 0:
        return None

    item = random.choice(items)
    tmp = {k:v[:bs] for k, v in mul_scale_datas[item].items()}
    mul_scale_datas[item] = {k:v[bs:] for k, v in mul_scale_datas[item].items() if isinstance(v, torch.Tensor)}
    mul_scales_count[item] -= bs
    return tmp


mul_scale_datas = []
mul_scales = []
mul_scales_count = []

def main():
    """测试不resize的多batchsize的dataloader
    """
    root_path = '/root/Source/ymy_dataset/Linemod_preprocessed'
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    dataset = PoseDataset('train', 500, True, root_path, 0.000, 8, cls_type='ape')

    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=10, collate_fn=my_colla_fn)
    ds_iter = dataloader_test.__iter__()

    for _ in range(2000):
        datas = None
        while datas is None:
            patch_datas = ds_iter.__next__()
            datas = process_patch_datas(patch_datas, 128)
        print(datas['img_croped'].shape)

if __name__ == '__main__':
    main()