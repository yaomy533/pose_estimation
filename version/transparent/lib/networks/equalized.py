#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/12/20 14:07
# @Author  : yaomy
import torch
from torch import nn
import math

_initial_missing = object()


def reduce(function, sequence, initial=_initial_missing):
    """
    reduce(function, sequence[, initial]) -> value

    Apply a function of two arguments cumulatively to the items of a sequence,
    from left to right, so as to reduce the sequence to a single value.
    For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
    ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
    of the sequence in the calculation, and serves as a default when the
    sequence is empty.
    """

    it = iter(sequence)

    if initial is _initial_missing:
        try:
            value = next(it)
        except StopIteration:
            raise TypeError("reduce() of empty sequence with no initial value") from None
    else:
        value = initial

    for element in it:
        value = function(value, element)

    return value


def mul(a, b):
    """Same as a * b."""
    return a * b


class Equalized(nn.Module):

    def __init__(self, module, equalized=True, lr_scale=1.0, bias=True):
        r"""
        equalized (bool): if True use He's constant to normalize at runtime.
        bias_zero_init (bool): if true, bias will be initialized to zero
        """
        super().__init__()

        assert module.bias is None
        self.module = module
        self.equalized = equalized

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.module.out_channels))

        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lr_scale
            self.weight = self.get_he_constant() * lr_scale

    def forward(self, x):
        x = self.module(x)
        if self.equalized:
            x *= self.weight
        dims = [1 for _ in x.shape]
        dims[1] = -1
        x = x + self.bias.view(*dims)
        return x

    def get_he_constant(self):
        r"""
        Get He's constant for the given layer
        https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
        """
        size = self.module.weight.size()
        fan_in = reduce(mul, size[1:], 1)

        return math.sqrt(2.0 / fan_in)  # Kaiming 初始化


class EqualizedConv2d(Equalized):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding: int = 0, padding_mode='zeros', **kwargs):
        module = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False,
                           padding=padding, padding_mode=padding_mode)
        super().__init__(module, **kwargs)


class EqualizedConv1d(Equalized):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding: int = 0, padding_mode='zeros', **kwargs):
        module = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, bias=False,
                           padding=padding, padding_mode=padding_mode)
        super().__init__(module, **kwargs)
