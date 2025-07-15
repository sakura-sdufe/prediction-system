# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/30 14:49
# @Author   : 张浩
# @FileName : cnn.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import torch
import numpy as np
from torch import nn
from base import BaseRegression, get_activation_fn, get_activation_nn


def calculate_output_size(input_size, kernel_size, stride=1, padding=0):
    """
    计算卷积层的输出大小。
    :param input_size: 输入大小。
    :param kernel_size: 卷积核大小。
    :param stride: 步长。
    :param padding: 填充大小。
    :return: 输出大小。
    Note:
        特殊地，在 padding=kernel_size//2、stride=1 的条件下，
            若 input_size 为偶数、kernel_size 为奇数，则输出大小为 input_size//2；
            其他情况下，则输出大小为 input_size//2+1。
    """
    return (input_size - kernel_size + 2 * padding) // stride + 1


class CA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation='relu'):
        super(CA, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        return self.activation((self.conv(x)))


class CBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation='relu'):
        super(CBA, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), activation='relu', shortcut=True, width_factor=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(out_channels * width_factor)
        self.conv1 = CBA(in_channels, c_, kernel_size=kernel_size[0], stride=1, padding=kernel_size[0]//2,
                         activation=activation)
        self.conv2 = CBA(c_, out_channels, kernel_size=kernel_size[1], stride=1, padding=kernel_size[1]//2,
                         activation=activation)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x if self.add else self.conv2(self.conv1(x))


class C2L(BaseRegression):
    def __init__(self, input_size, output_size, activation='relu'):
        super(C2L, self).__init__()
        self.conv1 = CA(1, 2, kernel_size=3, stride=2, padding=1, activation=activation)
        linear_in = calculate_output_size(input_size, kernel_size=3, stride=2, padding=1)
        self.conv2 = CA(2, 4, kernel_size=3, stride=2, padding=1, activation=activation)
        linear_in = calculate_output_size(linear_in, kernel_size=3, stride=2, padding=1)

        self.linear = nn.Linear(linear_in * 4, output_size)

    def forward(self, x):
        """
        1. 输入应当是一个 Tensor，且维度应为：[batch_size, input_size]；
        2. 输出应当是一个 Tensor，且维度应为：[batch_size, output_size]。
        """
        x = self.conv2(self.conv1(x.unsqueeze(1)))  # shape: [batch_size, channels, linear_in]
        return self.linear(x.reshape(x.shape[0], -1))  # shape: [batch_size, output_size]
