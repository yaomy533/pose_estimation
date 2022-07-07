#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 20:41
# @Author  : yaomy
import shutil
import os


def clrdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
