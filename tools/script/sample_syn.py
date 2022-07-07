#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/8 11:52
# @Author  : yaomy
####################
# 从渲染的数据里面每个类型每个物体随机采样1000个，也就是每个物体2000个
####################
import os
from pathlib import Path
import random

root = '/root/Source/ymy_dataset/Linemod_preprocessed'

obj_dict = {
            'ape': 1, 'benchvise': 2, 'cam': 4, 'can': 5, 'cat': 6, 'driller': 8,
            'duck': 9, 'eggbox': 10, 'glue': 11, 'holepuncher': 12, 'iron': 13, 'lamp': 14, 'phone': 15,
}

def read_lines(p):
    with open(p, 'r') as f:
        return [line.strip() for line in f.readlines()]

for obj in obj_dict.keys():
    for tp in ['renders', 'fuse']:
        all_items = read_lines(os.path.join(root, tp, obj, 'file_list.txt'))
        items = sorted(random.sample(all_items, 5000))
        with open(os.path.join(root, tp, obj, 'file_list_part_5000.txt'), 'w') as fp:
            for line in items:
                fp.write(line)
                fp.write('\n')  # 显示写入换行

