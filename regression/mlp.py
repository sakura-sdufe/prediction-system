# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/30 09:37
# @Author   : 张浩
# @FileName : mlp.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import torch.nn as nn
from typing import Sequence

from base import BaseRegression, get_activation_fn


class MLP(BaseRegression):
    def __init__(self, input_size, output_size, *, hidden_sizes:Sequence[int]=None, activation='relu',
                 dropout=0.1):
        """
        初始化多层感知机模型。输入层节点个数为：input_size，输出层节点个数为：output_size。
        :param input_size: 输入特征维度。
        :param output_size: 输出特征维度。
        :param hidden_sizes: 隐藏层维度。默认值为 None，表示直接从输入层映射到输出层。如果内部元素取值为 0，表示该位置采用残差连接。
        :param activation: 激活函数，可选 'relu'、'gelu'。默认值为 'relu'。
        :param dropout: dropout 概率。默认值为 0.1。
        :note: 如果 hidden_sizes 需要使用残差，请不要把 0 放在第一个位置，这是没有意义的。
        """
        super(MLP, self).__init__()
        if hidden_sizes is None:
            hidden_sizes = []
        assert hidden_sizes[0] > 0, "hidden_sizes 的第一个元素必须大于 0，因为输入层的维度是 input_size*time_step。"

        previous_size = input_size
        layers, linear_idx, residual_idx, residual_start = {}, 0, 0, previous_size  # 层字典、线性层索引、残差连接索引、残差连接输入维度
        for hidden_idx, hidden_size in enumerate(hidden_sizes):
            if hidden_size == 0:
                if residual_start == hidden_sizes[hidden_idx-1]:
                    layers[f'res {residual_idx}->{linear_idx - 1}'] = nn.Identity()
                else:
                    layers[f'res {residual_idx}->{linear_idx-1}'] = nn.Linear(residual_start, hidden_sizes[hidden_idx-1])
                residual_start = hidden_sizes[hidden_idx-1]  # 更新残差连接的输入维度
                residual_idx = linear_idx  # 更新残差连接的索引
            elif hidden_size > 0:
                layers[f'linear {linear_idx}'] = nn.Linear(previous_size, hidden_size)
                linear_idx += 1  # 更新线性层索引
                previous_size = hidden_size  # 更新线型输入维度
            else:
                raise ValueError(f"hidden_size 不能为负数，当前值为：{hidden_size}。如果您想使用残差连接，请将 hidden_size 设置为 0。")

        self.hidden_layers = nn.ModuleDict(layers)
        self.output_projection = nn.Linear(previous_size, output_size)

        self.activation = get_activation_fn(activation)  # function
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        前向传播。
        :param X: 输入张量，维度为 [batch_size, input_size]。
        :return: 输出张量，维度为 [batch_size, outputs_size]。
        """
        Y = X.clone()  # 保留输入张量，用于残差连接
        for key, layer in self.hidden_layers.items():
            if 'res' in key:
                Y = layer(X) + Y
                X = Y.clone()
            elif 'linear' in key:
                Y = self.dropout(self.activation(layer(Y)))
            else:
                raise ValueError(r"hidden_layers 的 filename 必须包含 'res' 或 'linear'。")
        Y = self.output_projection(Y)
        return Y
