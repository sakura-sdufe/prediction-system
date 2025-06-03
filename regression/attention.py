# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/30 09:48
# @Author   : 张浩
# @FileName : attention.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from torch import nn
from base import BaseRegression, get_activation_fn, get_activation_nn


class CAttention(nn.Module):
    def __init__(self, embed_dim=8, num_heads=2, dropout=0.1, bias=True):
        """
        添加卷积多通道的多头注意力机制。
        :param embed_dim: 嵌入维度。
        :param num_heads: 多头注意力的头数。
        :param dropout: dropout 概率。默认值为 0.1。（删的是注意力）
        :param bias: 多头注意力是否使用偏置。默认值为 True。
        """
        super(CAttention, self).__init__()
        self.attn_weight = None  # 保存注意力权重
        self.attn_output = None  # 保存注意力输出

        self.q_conv = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        self.k_conv = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        self.v_conv = nn.Conv1d(in_channels=1, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv1d(in_channels=embed_dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias)

    def forward(self, q, k ,v):
        """
        前向传播
        :param q: 查询张量，维度为 [batch_size, dim_q]。
        :param k: 键张量，维度为 [batch_size, dim_k]。
        :param v: 值张量，维度为 [batch_size, dim_k]。
        :return: 输出张量，维度为 [batch_size, dim_q]。
        """
        q, k, v = self.q_conv(q.unsqueeze(1)), self.k_conv(k.unsqueeze(1)), self.v_conv(v.unsqueeze(1))  # shape: [batch_size, embed_dim, dim_q]
        q, k, v = q.permute(2, 0, 1), k.permute(2, 0, 1), v.permute(2, 0, 1)  # shape: [dim_q, batch_size, embed_dim]
        self.attn_output, self.attn_weight = self.attn(q, k, v)  # shape: [dim_q, batch_size, embed_dim]
        self.attn_output = self.output_conv(self.attn_output.permute(1, 2, 0)).squeeze(1)  # shape: [batch_size, dim_q]
        return self.attn_output, self.attn_weight


class CAttn(BaseRegression):
    def __init__(self, input_size, output_size, *, embed_dim=8, num_heads=2, dropout=0.1, bias=True, activation='relu'):
        """
        将多个特征使用多头注意力机制进行融合。注意：这个过程无时序性，也没有位置编码，因此不能用于预测。
        :param input_size: 输入的维度。
        :param output_size: 输出的维度。
        :param embed_dim: 嵌入维度。
        :param num_heads: 多头注意力的头数。
        :param dropout: dropout 概率。默认值为 0.1。
        :param bias: 多头注意力是否使用偏置。默认值为 True。
        :param activation: 激活函数。默认值为 'relu'，可选值 'relu', 'gelu'。
        """
        super(CAttn, self).__init__()
        self.attn_weight = None  # 保存注意力权重
        self.attention = CAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=bias)
        self.output_project = nn.Linear(input_size, output_size)
        self.activation = get_activation_fn(activation)  # 激活函数

    def forward(self, X):
        """
        前向传播
        :param X: 输入张量，维度为 [batch_size, input_size]。
        :return: 输出张量，维度为 [batch_size, output_size]。
        """
        attn_output, self.attn_weight = self.attention(X, X, X)
        attn_output = self.activation(attn_output + X)  # shape: [batch_size, input_size]
        output = self.output_project(attn_output)  # shape: [batch_size, output_size]
        return output


class CAttnProj(BaseRegression):
    def __init__(self, input_size, output_size, *, embed_dim=8, num_heads=2, project_size=256,
                 feedforward=2048, dropout=0.1, bias=True, activation='relu'):
        """
        将多个特征使用具有高维映射的 Attention 进行融合。注意：这个过程无时序性，也没有位置编码，因此不能用于预测。
        :param input_size: 输入的维度。
        :param output_size: 输出的维度。
        :param embed_dim: 嵌入维度。
        :param num_heads: 多头注意力的头数。
        :param project_size: 参与注意力计算的特征维度。
        :param feedforward: 前馈神经网络的隐藏层维度。
        :param dropout: dropout 概率。默认值为 0.1。
        :param bias: 多头注意力是否使用偏置。默认值为 True。
        :param activation: 激活函数。默认值为 'relu'，可选值 'relu', 'gelu'。
        """
        super(CAttnProj, self).__init__()
        self.attn_weight = None  # 保存注意力权重
        self.activation = get_activation_fn(activation)  # 激活函数

        self.input_project = nn.Linear(input_size, project_size)  # 将特征维度映射到嵌入维度
        self.attention = CAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, bias=bias)  # 注意力机制
        self.output_project = nn.Sequential(
            nn.Linear(project_size, feedforward),
            get_activation_nn(activation),
            nn.Linear(feedforward, output_size),
        )  # 将特征维度映射到输出层维度层

    def forward(self, X):
        """
        前向传播
        :param X: 输入张量，维度为 [batch_size, input_size]。
        :return: 输出张量，维度为 [batch_size, output_size]。
        """
        X = self.activation(self.input_project(X))  # shape: [batch_size, project_size]
        attn_output, self.attn_weight = self.attention(X, X, X)
        attn_output = self.activation(attn_output + X)  # shape: [batch_size, project_size]
        output = self.output_project(attn_output)  # shape: [batch_size, output_size]
        return output
