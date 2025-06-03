# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/30 15:53
# @Author   : 张浩
# @FileName : transformer.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import torch
import torch.nn as nn
from typing import Sequence

from regression.mlp import MLP
from base import BaseSequence, RepeatLayer, get_activation_fn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_length=1000, dtype=torch.float32):
        """
        初始化位置编码，为位置编码创建正弦模式。
        Args:
            model_dim: 输入的特征维度，也等于输出的特征维度。
            max_length: 最大长度，默认值为 1000。
            dtype: 数据类型，默认值为 torch.float32。
        Note:
            位置编码矩阵的维度为 [1, max_length, model_dim] == [batch_size, time_step, model_dim]。
        """
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim
        # 位置编码矩阵
        pe = self.angle_defn(
            pos = torch.arange(max_length, dtype=dtype).unsqueeze(1),
            i = torch.arange(model_dim, dtype=dtype).unsqueeze(0),
        )
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册为模型参数。需要得到梯度信息时，不会更新因为 optimizer 而更新。
        self.register_buffer('positional_encoding', pe)
        # self.positional_encoding = nn.Parameter(pe, requires_grad=True)  # 使用 nn.Parameter() 使得位置编码矩阵可训练。

    def angle_defn(self, pos, i):
        """
        定义角度
        :param pos: 位置（时间序列方向）
        :param i: 位置编码的维度（特征方向）
        :return: 角度
        """
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / self.model_dim)
        return pos * angle_rates

    def forward(self, X):
        """
        前向传播
        Args:
            X: 输入张量，维度为 [time_step, batch_size, model_dim]。
        Returns:
            位置编码后的张量，维度为 [time_step, batch_size, model_dim]。
        """
        X = X.permute(1, 0, 2)  # 转置，维度变为 [batch_size, time_step, model_dim]
        assert X.size(-1) == self.positional_encoding.size(-1), "输入维度和位置编码维度必须匹配"
        positional_result = X + self.positional_encoding[:, 0:X.size(1), :]
        return positional_result.permute(1, 0, 2)  # 转置，维度变为 [time_step, batch_size, model_dim]


class AttentionLinearLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        """
        初始化 AttentionLinearLayer 层，包含多头注意力机制和前馈神经网络。
        :param d_model: 模型维度，输入和输出的特征维度。
        :param nhead: 多头注意力机制的头数。
        :param dim_feedforward: 前馈神经网络的隐藏层维度，默认值为 2048。
        :param dropout: dropout 概率，默认值为 0.1。
        :param activation: 激活函数，默认值为 'relu'。可选值为 'relu'、'gelu'。
        """
        super(AttentionLinearLayer, self).__init__()
        # 多头注意力部分
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # 前馈神经网络部分
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, X):
        """
        前向传播
        :param X: 输入张量，维度为 [time_step, batch_size, d_model]
        :return: 输出张量，维度为 [time_step, batch_size, d_model]
        """
        # MultiheadAttention
        attn_output, _ = self.self_attn(X, X, X)
        X = X + self.dropout(self.activation(attn_output))
        X = self.norm1(X)
        # Feedforward
        linear_output = self.dropout(self.activation(self.linear1(X)))
        linear_output = self.dropout(self.activation(self.linear2(linear_output)))
        X = X + linear_output
        X = self.norm2(X)
        return X


class TranMLP(BaseSequence):
    def __init__(self, input_size, output_size, *,
                 encoder_model_dim=128, encoder_head_num=8, encoder_feedforward_dim=2048, encoder_layer_num=2,
                 decoder_hidden_sizes:Sequence[int]=None, activation='relu', dropout=0.1, max_length=1000):
        """
        初始化 TranMLP 预测模型。编码器为 TransformerEncoder，解码器为全连接层。
        :param input_size: 输入特征维度。
        :param output_size: 输出特征维度。
        :param encoder_model_dim: 编码器 TransformerEncoderLayer 模型维度，默认值为 128。
        :param encoder_head_num: 编码器 TransformerEncoderLayer 多头注意力机制的头数，默认值为 8。
        :param encoder_feedforward_dim: 编码器 TransformerEncoderLayer 前馈神经网络的隐藏层维度，默认值为 2048。
        :param encoder_layer_num: 编码器 TransformerEncoderLayer 层数，默认值为 2。
        :param decoder_hidden_sizes: 解码器全连接层的隐藏层维度列表，默认值为 None，表示直接映射到输出维度。
        :param activation: 编码器和解码器的激活函数，默认值为 'relu'。
        :param dropout: 编码器 TransformerEncoderLayer 和 解码器全连接层的 dropout 概率，默认值为 0.1。
        :param max_length: 位置编码的最大长度，默认值为 1000。主要用于位置编码。
        """
        super(TranMLP, self).__init__()
        if decoder_hidden_sizes is None:
            decoder_hidden_sizes = []
        # 输入映射 和 位置编码
        self.input_projection = nn.Linear(input_size, encoder_model_dim)  # 输入特征维度 -> 模型维度
        self.positional_encoding = PositionalEncoding(encoder_model_dim, max_length)  # 位置编码
        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_model_dim, nhead=encoder_head_num, dim_feedforward=encoder_feedforward_dim, dropout=dropout,
            activation=activation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layer_num)
        # 解码器
        self.decoder = MLP(input_size=encoder_model_dim, output_size=output_size, hidden_sizes=decoder_hidden_sizes,
                           activation=activation, dropout=dropout)

        self.activation = get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        前向传播
        :param X: 输入张量，维度为 [time_step, batch_size, input_size]
        :return: 输出张量，维度为 [batch_size, output_size]
        """
        X = self.input_projection(X)  # 输入映射
        X = self.positional_encoding(X)  # 位置编码
        X = self.encoder(X)  # 编码器，由 TransformerEncoder 实现
        Y = X[-1, :, :]  # 取出最后一个时间步的隐藏层状态，作为全连接层的输入。
        Y = self.decoder(Y)
        return Y


class TranAttnFNN(BaseSequence):
    def __init__(self, input_size, output_size, *,
                 encoder_model_dim=128, encoder_head_num=8, encoder_feedforward_dim=2048, encoder_layer_num=2,
                 decoder_model_dim=128, decoder_head_num=8, decoder_feedforward_dim=2048, decoder_layer_num=2,
                 activation='relu', dropout=0.1, max_length=1000):
        """
        初始化 TranAttnFNN 预测模型。编码器为 TransformerEncoder，解码器为 AttentionLinearLayer（包含多头注意力机制和前馈神经网络）。
        :param input_size: 输入特征维度。
        :param output_size: 输出特征维度。
        :param encoder_model_dim: 编码器 TransformerEncoderLayer 模型维度，默认值为 128。
        :param encoder_head_num: 编码器 TransformerEncoderLayer 多头注意力机制的头数，默认值为 8。
        :param encoder_feedforward_dim: 编码器 TransformerEncoderLayer 前馈神经网络的隐藏层维度，默认值为 2048。
        :param encoder_layer_num: 编码器 TransformerEncoderLayer 层数，默认值为 2。
        :param decoder_model_dim: 解码器 MultiHeadAttention 模型维度，默认值为 128。
        :param decoder_head_num: 解码器 MultiHeadAttention 多头注意力机制的头数，默认值为 8。
        :param decoder_feedforward_dim: 解码器 MultiHeadAttention 前馈神经网络的隐藏层维度，默认值为 2048。
        :param decoder_layer_num: 解码器 MultiHeadAttention 层数，默认值为 2。
        :param activation: 编码器和解码器的激活函数，默认值为 'relu'。
        :param dropout: 编码器 TransformerEncoderLayer 和 解码器全连接层的 dropout 概率，默认值为 0.1。
        :param max_length: 位置编码的最大长度，默认值为 1000。主要用于位置编码。
        """
        super(TranAttnFNN, self).__init__()
        # 输入映射 和 位置编码
        self.input_project = nn.Linear(input_size, encoder_model_dim)  # 输入特征维度 -> 模型维度
        self.positional_encoding = PositionalEncoding(encoder_model_dim, max_length)  # 位置编码
        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_model_dim, nhead=encoder_head_num, dim_feedforward=encoder_feedforward_dim,
            dropout=dropout, activation=activation
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layer_num)
        # 编码器 -> 解码器
        self.transition = nn.Linear(encoder_model_dim, decoder_model_dim)
        # 解码器
        decoder_layer = AttentionLinearLayer(
            d_model=decoder_model_dim, nhead=decoder_head_num, dim_feedforward=decoder_feedforward_dim,
            dropout=dropout, activation=activation
        )
        self.decoder = RepeatLayer(decoder_layer, decoder_layer_num)
        # 解码器 -> 输出
        self.output_project = nn.Linear(decoder_model_dim, output_size)

    def forward(self, X):
        """
        前向传播
        :param X: 输入张量，维度为 [time_step, batch_size, input_size]
        :return: 输出张量，维度为 [batch_size, output_size]
        """
        X = self.input_project(X)
        X = self.positional_encoding(X)
        X = self.encoder(X)
        X = self.transition(X)
        X = self.decoder(X)
        Y = X[-1, :, :]
        Y = self.output_project(Y)
        return Y
