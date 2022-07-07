#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 23:04
# @Author  : yaomy
import random
import torch
import numpy as np
from torch.optim.lr_scheduler import CyclicLR, StepLR
from lib.network.optimizer.ranger import flat_and_anneal_lr_scheduler

######################################################
'''
    The follows are copy from pytorch
'''

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
######################################################


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


def build_lr_scheduler(arg, optimizer):
    cfg = arg.cfg
    if cfg.Train.Lr.LR_SCHEDULER == 'epoch':
        print('Using Lr Scheduler of StepLR!')
        lr_scheduler = StepLR(optimizer, step_size=cfg.Train.Lr.EPOCH.STEP_SIZE, gamma=cfg.Train.Lr.EPOCH.GAMMA)
    elif cfg.Train.Lr.LR_SCHEDULER == 'lambda':
        print('Using Lr Scheduler of Lambda!')
        total_iters = arg.deacline_step / cfg.Train.Lr.LAMBDA.ANNEAL_POINT
        lr_scheduler = flat_and_anneal_lr_scheduler(
            optimizer,
            total_iters=total_iters,  # NOTE: TOTAL_EPOCHS * len(train_loader)
            warmup_factor=cfg.Train.Lr.LAMBDA.WARMUP_FACTOR,
            warmup_iters=cfg.Train.Lr.LAMBDA.WARMUP_ITERS,
            warmup_method=cfg.Train.Lr.LAMBDA.WARMUP_METHOD,  # default "linear"
            anneal_method=cfg.Train.Lr.LAMBDA.ANNEAL_METHOD,
            anneal_point=cfg.Train.Lr.LAMBDA.ANNEAL_POINT,  # default 0.72
            steps=[2 / 3.0, 8 / 9.0],  # default [2/3., 8/9.], relative decay steps
            target_lr_factor=0,
            poly_power=1.0,
            step_gamma=cfg.Train.Lr.LAMBDA.GAMMA,  # default 0.1
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def worker_init_fn(worker_id):
    """固定每一次epoch的顺序，以及每个epoch的顺序，但是各个epoch之间不相同
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)
