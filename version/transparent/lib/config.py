#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 13:49
# @Author  : yaomy

def cleargrasp_parameter(opt):
    # output root dir
    output_root = '/root/Source/ymy_dataset/trained/instance/cleargrasp'
    opt.decay_start = False
    opt.refine_start = False
    opt.num_points = 1000
    opt.num_objects = 5

    # logdir
    opt.outf = f'{output_root}/{opt.log_file}'  # folder to save trained models
    opt.log_dir = f'{output_root}/{opt.log_file}'  # folder to save logs

    return opt
