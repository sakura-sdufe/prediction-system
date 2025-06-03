# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/5/9 16:28
# @Author   : 张浩
# @FileName : main.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from runner import Runner
from utils.criterion import Huber_loss, sMAPELoss, QuantileLoss

from regression import SVR, GradientBoostingRegressor, AdaBoostRegressor, Ridge, RandomForestRegressor, BaggingRegressor
from regression import MLP, CAttn, CAttnProj
from sequence import RNN, LSTM, GRU, TranMLP, TranAttnFNN


if __name__ == '__main__':
    # 读取参数
    data_parameter_path = r"./parameters/data.yaml"
    model_parameter_path = r"./parameters/model2.yaml"
    # 声明损失函数和监控函数
    criterion = Huber_loss
    # criterion = QuantileLoss
    monitor = sMAPELoss
    # 选择模型
    models_onestep = [SVR, GradientBoostingRegressor, AdaBoostRegressor]  # 仅支持单个目标单步预测
    normalizations_onestep = [True, True, True]  # models_onestep 模型是否归一化
    models_multistep = [Ridge, RandomForestRegressor, BaggingRegressor, MLP, CAttn, CAttnProj, RNN, LSTM, GRU,
                        TranMLP, TranAttnFNN]  # 支持多目标或多步预测
    normalizations_multistep = [True, False, False, True, True, True, True, True, True, True, True]  # models_multistep 模型是否归一化
    models = models_onestep + models_multistep
    normalizations = normalizations_onestep + normalizations_multistep
    # 启动预测
    runner = Runner(data_parameter_path=data_parameter_path, model_parameter_path=model_parameter_path)
    runner.run(models, is_normalization=normalizations, criterion=criterion, monitor=monitor)
