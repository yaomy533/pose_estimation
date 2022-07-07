#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# 2021/3/12 下午1:48
# author:yaomy


# cls_list = [
#     'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone'
# ]
cls_list = [
    'duck'
]
for cls in cls_list:
    path = '/home/public/linemod/Linemod_preprocessed/fuse/{}/file_list.txt'.format(cls)
    f = open(path, 'r+')
    new_lines = []
    lines = f.readlines()
    print(len(lines))
    for line in lines:
        line = line.split('/L')
        newline = '/home/public/linemod/L{}'.format(line[1])
        new_lines.append(newline)
    f.seek(0)
    f.truncate()
    print(len(new_lines))
    f.writelines(new_lines)
    f.close()

    ff = open(path, 'r+')
    llines = ff.readlines()
    print(len(llines))
