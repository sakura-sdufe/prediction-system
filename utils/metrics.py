# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/29 15:17
# @Author   : 张浩
# @FileName : metrics.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import mean_squared_log_error, median_absolute_error, max_error, r2_score


def convert_to_numpy(data):
    # 转换数据类型
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.to_numpy()
    elif isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise ValueError("转换类型失败！")
    data = data.astype(float)  # 转换为 float 类型
    # 转换为 2 维
    if data.ndim == 1:
        warn_info_dim = "由于一维数据无法判断该维度是属于样本维度（样本个数），还是输出维度（输出个数），默认将其认作样本维度。如果传入的" + \
                        "数据类型为 pd.Series 建议转为 pd.DataFrame，如果传入的数据类型为 np.ndarray 或 list，建议传入二位数组或列表。"
        warnings.warn(warn_info_dim)
        data = np.expand_dims(data, axis=1)
    elif data.ndim > 2:
        raise ValueError("维度转换失败，只接受维度大小为 1 和 2！")
    return data


def weights_to_ndarray(weights) -> np.ndarray:
    """将 weights 转换为 1 维的 np.ndarray，支持 list 和 1 维 np.ndarray。"""
    if isinstance(weights, list):
        weights = np.array(weights)
    assert np.all(weights > 0) and weights.ndim == 1, "权重值必须大于 0，并且权重维度只能为 1！"
    if not np.allclose(sum(weights), 1.0, rtol=1e-07, atol=1e-08):
        warnings.warn(f"输入的权重总和应当为 1.0，但实际为 {sum(weights)}，已自动放缩到总和为 1.0！")
        weights = weights / np.sum(weights)  # 归一化
    return weights


def MSE(true_value, predict_value, weights):
    """均方误差（Mean Squared Error）"""
    mse_value = 0.0
    for ind, weight in enumerate(weights):
        mse_value += mean_squared_error(true_value[:, ind], predict_value[:, ind]) * weight
    return mse_value


def RMSE(true_value, predict_value, weights):
    """均方根误差（Root Mean Squared Error）"""
    return np.sqrt(MSE(true_value, predict_value, weights))


def MAE(true_value, predict_value, weights):
    """均值绝对误差（Mean Absolute Error）"""
    mae_value = 0.0
    for ind, weight in enumerate(weights):
        mae_value += mean_absolute_error(true_value[:, ind], predict_value[:, ind]) * weight
    return mae_value


def MAPE(true_value, predict_value, weights):
    """平均绝对百分比误差（Mean Absolute Percentage Error）"""
    mape_value = 0.0
    for ind, weight in enumerate(weights):
        mape_value += mean_absolute_percentage_error(true_value[:, ind], predict_value[:, ind]) * weight
    return mape_value


def SMAPE(true_value, predict_value, weights):
    """对称平均绝对百分比误差（Symmetric Mean Absolute Percentage Error）"""
    epsilon = 1e-6  # 防止分母为 0
    Numerator = np.abs(true_value - predict_value)
    Denominator = (np.abs(true_value) + np.abs(predict_value)) / 2 + epsilon
    return np.sum(np.mean(Numerator / Denominator, axis=0) * weights)


def MSLE(true_value, predict_value, weights):
    """均方对数误差（Mean Squared Logarithmic Error）"""
    msle_value = 0.0
    for ind, weight in enumerate(weights):
        msle_value += mean_squared_log_error(true_value[:, ind], predict_value[:, ind]) * weight
    return msle_value


def RMSLE(true_value, predict_value, weights):
    """均方对数误差（Root Mean Squared Logarithmic Error）"""
    return np.sqrt(MSLE(true_value, predict_value, weights))


def MedAE(true_value, predict_value, weights):
    """中位数绝对误差（Median Absolute Error）"""
    medae_value = 0.0
    for ind, weight in enumerate(weights):
        medae_value += median_absolute_error(true_value[:, ind], predict_value[:, ind]) * weight
    return medae_value


def MaxError(true_value, predict_value, weights):
    """最大绝对误差（Max Absolute Error）"""
    maxerror_value = 0.0
    for ind, weight in enumerate(weights):
        maxerror_value += max_error(true_value[:, ind], predict_value[:, ind]) * weight
    return maxerror_value


def RAE(true_value, predict_value, weights):
    """相对绝对误差（Relative Absolute Error）"""
    Numerator = np.sum(np.abs(true_value - predict_value), axis=0)
    Denominator = np.sum(np.abs(true_value - np.mean(true_value, axis=0)), axis=0)
    return np.sum(Numerator / Denominator * weights)


def RSE(true_value, predict_value, weights):
    """相对平方误差（Relative Squared Error）"""
    Numerator = np.sum(np.square(true_value - predict_value), axis=0)
    Denominator = np.sum(np.square(true_value - np.mean(true_value, axis=0)), axis=0)
    return np.sum(Numerator / Denominator * weights)


def Huber(true_value, predict_value, weights, delta=1.0):
    """Huber 损失"""
    difference = np.abs(true_value - predict_value)
    mse_position = difference <= delta
    difference[mse_position] = 0.5 * np.square(difference[mse_position])
    difference[~mse_position] = delta * (difference[~mse_position] - 0.5 * delta)
    huber = np.mean(difference, axis=0)
    return np.sum(huber * weights)


def R2(true_value, predict_value, weights):
    r2_value = 0.0
    for ind, weight in enumerate(weights):
        r2_value += r2_score(true_value[:, ind], predict_value[:, ind]) * weight
    return r2_value


def calculate_metrics(true_value, predict_value, metrics=None, weights=None):
    """
    计算评价指标。
    :param true_value: 真实值。支持 Series，DataFrame，numpy.ndarray，list。
    :param predict_value: 预测值。支持 Series，DataFrame，numpy.ndarray，list。
    :param metrics: 评价指标。支持 "MSE"，"RMSE"，"MAE"，"MAPE"，"SMAPE"，"MSLE"，"MedAE"，"MaxError"，"MBE"，"RAE"，"RSE"，"RMSLE"，"Huber"，"R2"。
    :param weights: 不同列（不同输出）的评价指标占比。如果为 None，则表示所有列的评价指标占比一致。支持 list 和 1维 np.ndarray
    :return: 以字典的形式返回评价指标值。
    Note:
        1. true_value 和 predict_value 类型最高支持 2 维（不建议传入 1 维数据，可能会引起歧义）
        2. 如果需要添加新的评价指标，那么输入的 true_value 和 predict_value 均为 2D ndarray，且行数和列数均相同。
    """
    if metrics is None:
        metrics = ["MSE", "RMSE", "MAE", "MAPE", "SMAPE", "MSLE", "RMSLE", "MedAE", "MaxError", "RAE", "RSE", "Huber", "R2"]
    if isinstance(true_value, pd.DataFrame) and isinstance(predict_value, pd.DataFrame):
        assert predict_value.shape == true_value.shape, "输入的真实值和预测值的尺寸不匹配！"
        predict_value = predict_value[true_value.columns]  # 保证预测值和真实值的列名一致
    true_value = convert_to_numpy(true_value)  # 转换为 numpy 数组，并设置为 float 类型
    predict_value = convert_to_numpy(predict_value)  # 转换为 numpy 数组，并设置为 float 类型

    assert true_value.shape == predict_value.shape, "输入的真实值尺寸和预测值尺寸不匹配！"
    if weights is None:
        weights = [1/true_value.shape[1]] * true_value.shape[1]
    weights = weights_to_ndarray(weights)  # 将权重转为1维数组
    assert true_value.shape[1] == len(weights), f"权重个数为 {len(weights)} 与输出个数 {true_value.shape[1]} 不匹配！"

    metrics_function = {
        "MSE": MSE,
        "RMSE": RMSE,
        "MAE": MAE,
        "MAPE": MAPE,
        "SMAPE": SMAPE,
        "MSLE": MSLE,
        "RMSLE": RMSLE,
        "MedAE": MedAE,
        "MaxError": MaxError,
        "RAE": RAE,
        "RSE": RSE,
        "Huber": Huber,
        "R2": R2,
    }

    metrics_result = {}
    for metric in metrics:
        if metric in metrics_function.keys():
            metrics_result[metric] = metrics_function[metric](true_value, predict_value, weights)
        else:
            raise ValueError(f"不支持的评价指标：{metric}！")

    return metrics_result
