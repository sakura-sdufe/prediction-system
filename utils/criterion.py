# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/29 15:35
# @Author   : 张浩
# @FileName : criterion.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import numbers
import torch
from torch import Tensor
from torch import nn
import numpy as np
from typing import Union
import warnings


def weights_to_tensor(weights) -> Union[None, torch.Tensor]:
    """将 weights 转换为 1 维的 torch.tensor，支持 list、tuple、np.ndarray 和 torch.Tensor。"""
    if weights is None:
        return None
    elif isinstance(weights, (list, tuple)) and isinstance(weights[0], numbers.Number):
        weights = torch.tensor(weights, dtype=torch.float32)  # 1D list, tuple -> 1D torch.tensor
    elif isinstance(weights, np.ndarray) and weights.ndim == 1:
        weights = torch.from_numpy(weights).to(dtype=torch.float32)  # 1D np.ndarray -> 1D torch.tensor
    elif isinstance(weights, torch.Tensor) and weights.ndim == 1:
        weights = weights.to(dtype=torch.float32)  # 1D torch.tensor -> 1D torch.tensor
    else:
        raise ValueError(f"非法的损失函数权重值 {weights}。请使用 list、tuple、np.ndarray 或 torch.Tensor 类型的 1 维数值数组。")
    assert torch.all(weights > 0) and weights.ndim == 1, "损失函数权重值必须大于 0，并且权重维度只能为 1！"
    if not torch.allclose(weights.sum(), torch.tensor(1.0), rtol=1e-07, atol=1e-08):
        warnings.warn(f"损失函数权重总和应当为 1.0，但实际为 {weights.sum().cpu().float()}，已自动放缩到总和为 1.0！")
        weights = weights / torch.sum(weights)  # 归一化
    return weights


class Huber_loss(nn.HuberLoss):
    def __init__(self, weights=None):
        self.delta = 1.0  # Huber 损失函数的 delta 参数，控制损失函数的平滑程度。
        super(Huber_loss, self).__init__(reduction='none', delta=self.delta)
        self.weights = weights_to_tensor(weights)

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        """
        向前计算
        :param inputs: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :param targets: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :return: 损失值，数据类型为 torch.Tensor，形状为 [1]。
        """
        assert isinstance(inputs, Tensor) and inputs.ndim == 2, f"参数 input 的数据类型为 {type(inputs)}，但应为 torch.Tensor，且维度为 2！"
        assert isinstance(targets, Tensor) and targets.ndim == 2, f"参数 target 的数据类型为 {type(targets)}，但应为 torch.Tensor，且维度为 2！"
        assert inputs.shape == targets.shape, f"参数 input 的尺寸为 {inputs.shape}，但参数 target 的尺寸为 {targets.shape}，请保持一致！"
        if self.weights is None:
            self.weights = torch.ones(inputs.shape[-1], dtype=torch.float32) / inputs.shape[-1]  # 初始化一次
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(device=inputs.device)  # 转到指定设备
        # 计算损失值
        loss_value = torch.sum(super(Huber_loss, self).forward(inputs, targets).mean(dim=0) * self.weights)
        return loss_value


class MSELoss(nn.MSELoss):
    def __init__(self, weights=None):
        super(MSELoss, self).__init__(reduction='none')
        self.scale = 1.0
        self.power = 1.0
        self.weights = weights_to_tensor(weights=weights)

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        """
        向前计算
        :param inputs: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :param targets: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :return: 损失值，数据类型为 torch.Tensor，形状为 [1]。
        """
        assert isinstance(inputs, Tensor) and inputs.ndim == 2, f"参数 input 的数据类型为 {type(inputs)}，但应为 torch.Tensor，且维度为 2！"
        assert isinstance(targets, Tensor) and targets.ndim == 2, f"参数 target 的数据类型为 {type(targets)}，但应为 torch.Tensor，且维度为 2！"
        assert inputs.shape == targets.shape, f"参数 input 的尺寸为 {inputs.shape}，但参数 target 的尺寸为 {targets.shape}，请保持一致！"
        if self.weights is None:
            self.weights = torch.ones(inputs.shape[-1], dtype=torch.float32) / inputs.shape[-1]  # 初始化一次
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(device=inputs.device)  # 转到指定设备
        # 计算损失值
        mse_func = super(MSELoss, self).forward
        loss_value = torch.sum(
            torch.pow(mse_func(inputs, targets).mean(dim=0) * self.weights, self.power)
        ) * self.scale
        return loss_value


class sMAPELoss(nn.Module):
    def __init__(self, weights=None):
        super(sMAPELoss, self).__init__()
        self.epsilon = 1e-6  # 防止分母为 0
        self.weights = weights_to_tensor(weights)

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        """
        向前计算
        :param inputs: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :param targets: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :return: 损失值，数据类型为 torch.Tensor，形状为 [1]。
        """
        assert isinstance(inputs, Tensor) and inputs.ndim == 2, f"参数 input 的数据类型为 {type(inputs)}，但应为 torch.Tensor，且维度为 2！"
        assert isinstance(targets, Tensor) and targets.ndim == 2, f"参数 target 的数据类型为 {type(targets)}，但应为 torch.Tensor，且维度为 2！"
        assert inputs.shape == targets.shape, f"参数 input 的尺寸为 {inputs.shape}，但参数 target 的尺寸为 {targets.shape}，请保持一致！"
        if self.weights is None:
            self.weights = torch.ones(inputs.shape[-1], dtype=torch.float32) / inputs.shape[-1]  # 初始化一次
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(device=inputs.device)  # 转到指定设备
        # 计算损失值
        Numerator = torch.abs(targets - inputs)
        Denominator = (torch.abs(targets) + torch.abs(inputs)) / 2 + self.epsilon
        loss = (Numerator / Denominator)
        loss_value = torch.sum(loss.mean(dim=0) * self.weights)
        return loss_value


class MAPELoss(nn.Module):
    def __init__(self, weights=None):
        super(MAPELoss, self).__init__()
        self.epsilon = 1e-6  # 防止分母为 0
        self.weights = weights_to_tensor(weights)

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        """
        向前计算
        :param inputs: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :param targets: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :return: 损失值，数据类型为 torch.Tensor，形状为 [1]。
        """
        assert isinstance(inputs, Tensor) and inputs.ndim == 2, f"参数 input 的数据类型为 {type(inputs)}，但应为 torch.Tensor，且维度为 2！"
        assert isinstance(targets, Tensor) and targets.ndim == 2, f"参数 target 的数据类型为 {type(targets)}，但应为 torch.Tensor，且维度为 2！"
        assert inputs.shape == targets.shape, f"参数 input 的尺寸为 {inputs.shape}，但参数 target 的尺寸为 {targets.shape}，请保持一致！"
        if self.weights is None:
            self.weights = torch.ones(inputs.shape[-1], dtype=torch.float32) / inputs.shape[-1]  # 初始化一次
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(device=inputs.device)  # 转到指定设备
        # 计算损失值
        Numerator = torch.abs(targets - inputs)
        Denominator = torch.abs(targets) + self.epsilon
        loss = (Numerator / Denominator)
        loss_value = torch.sum(loss.mean(dim=0) * self.weights)
        return loss_value


class QuantileLoss(nn.Module):
    def __init__(self, weights=None):
        """分位数损失函数"""
        super(QuantileLoss, self).__init__()
        self.tau = 0.5  # 分位数参数，取值范围为 [0, 1]。
        assert 0 <= self.tau <= 1, f"分位数参数 tau 的取值范围为 [0, 1]，但实际为 {self.tau}！"
        self.weights = weights_to_tensor(weights)

    def forward(self, inputs:Tensor, targets:Tensor) -> Tensor:
        """
        向前计算
        :param inputs: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :param targets: 预测值，数据类型为 torch.Tensor，形状为 [batch_size, output_size]。
        :return: 损失值，数据类型为 torch.Tensor，形状为 [1]。
        """
        assert isinstance(inputs, Tensor) and inputs.ndim == 2, f"参数 input 的数据类型为 {type(inputs)}，但应为 torch.Tensor，且维度为 2！"
        assert isinstance(targets, Tensor) and targets.ndim == 2, f"参数 target 的数据类型为 {type(targets)}，但应为 torch.Tensor，且维度为 2！"
        assert inputs.shape == targets.shape, f"参数 input 的尺寸为 {inputs.shape}，但参数 target 的尺寸为 {targets.shape}，请保持一致！"
        if self.weights is None:
            self.weights = torch.ones(inputs.shape[-1], dtype=torch.float32) / inputs.shape[-1]  # 初始化一次
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(device=inputs.device)  # 转到指定设备
        # 计算损失值
        difference = targets - inputs
        loss = torch.where(difference >= 0, self.tau * difference, (self.tau - 1) * difference)
        loss_value = torch.sum(loss.mean(dim=0) * self.weights)
        return loss_value
