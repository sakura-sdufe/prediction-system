# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/6/4 16:54
# @Author   : 张浩
# @FileName : probabilistic_metrics.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import re
import os
import time
from typing import Union, Iterable
import numpy as np
import pandas as pd

from data import check_result_identify
from utils.metrics import convert_to_numpy, weights_to_ndarray
from utils.writer import Writer


def PICP(y_true, y_lower, y_upper, weights, **kwargs):
    """
    预测区间覆盖率（Prediction Interval Coverage Probability, PICP）。实际值落入预测区间的比例。
    :param y_true: 实际值，二维数组，形状为 (n_samples, n_outputs)。
    :param y_lower: 预测区间下限，二维数组，形状为 (n_samples, n_outputs)。
    :param y_upper: 预测区间上限，二维数组，形状为 (n_samples, n_outputs)。
    :param weights: 权重，1 维数组，长度为 n_outputs，表示每个输出的权重。
    :return: 预测区间覆盖率，浮点数，表示实际值落在预测区间内的比例。该值越接近 1.0，表示预测区间越可靠。
    """
    result = np.dot(np.mean((y_true >= y_lower) & (y_true <= y_upper), axis=0), weights)
    return result


def MPIW(y_lower, y_upper, weights, **kwargs):
    """
    平均区间宽度（Mean Prediction Interval Width, MPIW）。预测区间的平均宽度。
    :param y_lower: 预测区间下限，二维数组，形状为 (n_samples, n_outputs)。
    :param y_upper: 预测区间上限，二维数组，形状为 (n_samples, n_outputs)。
    :param weights: 权重，1 维数组，长度为 n_outputs，表示每个输出的权重。
    :return: 平均区间宽度，浮点数，表示所有预测区间的平均宽度。该值越小，表示预测区间越紧凑。
    """
    result = np.dot(np.mean(y_upper - y_lower, axis=0), weights)
    return result


def PINAW(y_true, y_lower, y_upper, weights, **kwargs):
    """
    区间归一化平均宽度（Prediction Interval Normalized Average Width, PINAW）。
    :param y_true: 实际值，二维数组，形状为 (n_samples, n_outputs)。
    :param y_lower: 预测区间下限，二维数组，形状为 (n_samples, n_outputs)。
    :param y_upper: 预测区间上限，二维数组，形状为 (n_samples, n_outputs)。
    :param weights: 权重，1 维数组，长度为 n_outputs，表示每个输出的权重。
    :return: 区间归一化平均宽度，浮点数，表示平均区间宽度与实际值范围的比率。该值越小，表示预测区间越紧凑。
    """
    scale = y_true.max(axis=0) - y_true.min(axis=0)  # 计算实际值范围
    result = np.dot(np.mean(y_upper - y_lower, axis=0) / scale, weights)  # 计算平均区间宽度归一化
    return result


def interval_score(y_true, y_lower, y_upper, weights, alpha, **kwargs):
    """
    区间分数（Interval Score, IS）。同时考虑区间宽度和是否包含真实值。
    :param y_true: 实际值，二维数组，形状为 (n_samples, n_outputs)。
    :param y_lower: 预测区间下限，二维数组，形状为 (n_samples, n_outputs)。
    :param y_upper: 预测区间上限，二维数组，形状为 (n_samples, n_outputs)。
    :param weights: 权重，1 维数组，长度为 n_outputs，表示每个输出的权重。
    :param alpha: 分位数，表示预测区间的置信水平。例如 0.05 表示 95% 的置信水平。
    :return: 区间分数，浮点数。该值越小，表示预测区间越可靠。
    """
    width = y_upper - y_lower  # 计算区间宽度
    under = (y_lower - y_true) * (y_true < y_lower)  # 计算下界未覆盖的惩罚
    over = (y_true - y_upper) * (y_true > y_upper)  # 计算上界未覆盖的惩罚
    score = width + (2 / alpha) * (under + over)  # 计算区间分数
    result = np.dot(np.mean(score, axis=0), weights)  # 计算加权平均区间分数
    return result


def CRPS(y_true, y_lower, y_upper, weights, **kwargs):
    """
    连续排名概率评分（Continuous Ranked Probability Score, CRPS）。Ref: Gneiting and Raftery (2004)
    :param y_true: 真实值，二维数组，形状为 (n_samples, n_outputs)。
    :param y_lower: 预测区间下限，二维数组，形状为 (n_samples, n_outputs)。
    :param y_upper: 预测区间上限，二维数组，形状为 (n_samples, n_outputs)。
    :param weights: 权重，1 维数组，长度为 n_outputs，表示每个输出的权重。
    :return: 连续排名概率评分，浮点数。该值越小，表示预测区间越可靠。
    """
    y_mid = (y_lower + y_upper) / 2  # 计算区间中点
    error = np.abs(y_true - y_mid)  # 计算实际值与中点的误差
    width = y_upper - y_lower  # 计算区间宽度
    result = np.dot(np.mean(error + 0.5 * width, axis=0), weights)  # 计算加权平均连续排名概率评分
    return result


def MAE_MidPoint(y_true, y_lower, y_upper, weights, **kwargs):
    """
    中心点绝对误差（Mean Absolute Error of Midpoint）。
    :param y_true: 真实值，二维数组，形状为 (n_samples, n_outputs)。
    :param y_lower: 预测区间下限，二维数组，形状为 (n_samples, n_outputs)。
    :param y_upper: 预测区间上限，二维数组，形状为 (n_samples, n_outputs)。
    :param weights: 权重，1 维数组，长度为 n_outputs，表示每个输出的权重。
    :return: 中心点绝对误差，浮点数。该值越小，表示预测区间越精确。
    """
    y_mid = (y_lower + y_upper) / 2  # 计算区间中点
    error = np.abs(y_true - y_mid)  # 计算实际值与中点的误差
    result = np.dot(np.mean(error, axis=0), weights)  # 计算加权平均中心点均方误差
    return result


def MSE_MidPoint(y_true, y_lower, y_upper, weights, **kwargs):
    """
    中心点均方误差（Mean Squared Error of Midpoint）。
    :param y_true: 真实值，二维数组，形状为 (n_samples, n_outputs)。
    :param y_lower: 预测区间下限，二维数组，形状为 (n_samples, n_outputs)。
    :param y_upper: 预测区间上限，二维数组，形状为 (n_samples, n_outputs)。
    :param weights: 权重，1 维数组，长度为 n_outputs，表示每个输出的权重。
    :return: 中心点均方误差，浮点数。该值越小，表示预测区间越精确。
    """
    y_mid = (y_lower + y_upper) / 2  # 计算区间中点
    error = (y_true - y_mid) ** 2  # 计算实际值与中点的平方误差
    result = np.dot(np.mean(error, axis=0), weights)  # 计算加权平均中心点均方误差
    return result


def CWC(y_true, y_lower, y_upper, weights, alpha, eta=10.0, **kwargs):
    """
    覆盖宽度标准（Coverage Width-based Criterion, CWC）。
    :param y_true: 真实值，二维数组，形状为 (n_samples, n_outputs)。
    :param y_lower: 预测区间下限，二维数组，形状为 (n_samples, n_outputs)。
    :param y_upper: 预测区间上限，二维数组，形状为 (n_samples, n_outputs)。
    :param weights: 权重，1 维数组，长度为 n_outputs，表示每个输出的权重。
    :param alpha: 分位数，表示预测区间的置信水平。例如 0.05 表示 95% 的置信水平。
    :param eta: 调节参数，控制覆盖率和宽度的平衡。默认值为 1.0。eta 越小，那么 PICP 没有达到置信水平的惩罚程度越小（重要程度越低）。
    :return: 覆盖宽度标准，浮点数。该值越小，表示预测区间越可靠。
    """
    u = 1 - alpha
    picp = PICP(y_true=y_true, y_lower=y_lower, y_upper=y_upper, weights=weights)  # 计算预测区间覆盖率
    pinaw = PINAW(y_true=y_true, y_lower=y_lower, y_upper=y_upper, weights=weights)  # 计算区间归一化平均宽度
    penalty = np.exp(-eta * (picp - u))  # 计算惩罚项
    result = pinaw * (1 + penalty if picp < u else 1)
    return result


def calculate_possibility(actual_value: pd.DataFrame, lower_value: pd.DataFrame, upper_value: pd.DataFrame, alpha: float,
                          metrics: Iterable=None, weights: Union[list, np.ndarray]=None):
    """
    计算概率预测评价指标。
    :param actual_value: 真实值。支持 Series, DataFrame, numpy.ndarray, list。
    :param lower_value: 下界预测结果。支持 Series, DataFrame, numpy.ndarray, list。
    :param upper_value: 上界预测结果。支持 Series, DataFrame, numpy.ndarray, list。
    :param alpha: 分位数，表示预测区间的置信水平。例如 0.05 表示 95% 的置信水平。
    :param metrics: 评价指标。支持 'PICP', 'MPIW', 'PINAW', 'interval_score', 'CRPS', 'MAE_MidPoint', 'MSE_MidPoint', 'CWC'。
    :param weights: 不同列（不同输出）的评价指标占比。如果为 None，则表示所有列的评价指标占比一致。支持 list 和 1 维 np.ndarray。
                    长度与 actual_value, lower_value 和 upper_value 的列数相同。
    Note:
        1. actual_value 和 lower_value, upper_value 必须具有相同的形状，最高支持 2 维（不建议传入 1 维数据，可能会引起歧义）
        2. 当 lower_value 大于 upper_value 时，会自动交换 lower_value 和 upper_value 的值。
        3. 如果需要添加新的评价指标，那么输入的 y_true, y_lower 和 y_upper 均为 2D ndarray，且行数和列数均相同。
    :return:
    """
    if metrics is None:
        metrics = ['PICP', 'MPIW', 'PINAW', 'interval_score', 'CRPS', 'MAE_MidPoint', 'MSE_MidPoint', 'CWC']
    if isinstance(actual_value, pd.DataFrame) and isinstance(lower_value, pd.DataFrame) and isinstance(upper_value, pd.DataFrame):
        assert actual_value.columns.equals(lower_value.columns) and actual_value.columns.equals(upper_value.columns), \
            "实际值、下界预测结果和上界预测结果的列名不一致！"
    actual_value = convert_to_numpy(actual_value)  # 转换为 numpy 数组，并设置为 float 类型
    lower_value = convert_to_numpy(lower_value)  # 转换为 numpy 数组，并设置为 float 类型
    upper_value = convert_to_numpy(upper_value)  # 转换为 numpy 数组，并设置为 float 类型
    assert lower_value.shape == upper_value.shape == actual_value.shape, "输入的下界、上界预测结果和实际值的尺寸不一致！"
    # 当下界值大于上界值时，交换下界和上界
    mask = lower_value > upper_value
    lower_value[mask], upper_value[mask] = upper_value[mask], lower_value[mask]  # 交换下界和上界

    if weights is None:
        weights = [1 / actual_value.shape[1]] * actual_value.shape[1]
    weights = weights_to_ndarray(weights)  # 将权重转为1维数组
    assert actual_value.shape[1] == len(weights), f"权重个数为 {len(weights)} 与输出个数 {actual_value.shape[1]} 不匹配！"

    metrics_function = {m: eval(m) for m in metrics}  # 获取评价指标函数
    metrics_result = {}
    for m, func in metrics_function.items():
        metrics_result[m] = func(y_true=actual_value, y_lower=lower_value, y_upper=upper_value, weights=weights, alpha=alpha)
    return metrics_result


def calculate_possibility_folds(lower_dir, upper_dir, save_dir, alpha, metrics: list=None, weights=None):
    """
    从指定的目录中读取预测结果，并且计算概率预测评价指标。
    :param lower_dir: 下界预测结果目录
    :param upper_dir: 上界预测结果目录
    :param save_dir: 保存结果的目录（要求必须为绝对地址）
    :param alpha: 分位数，表示预测区间的置信水平。例如 0.05 表示 95% 的置信水平。
    :param metrics: 评价指标。支持 'PICP', 'MPIW', 'PINAW', 'interval_score', 'CRPS', 'MAE_MidPoint', 'MSE_MidPoint', 'CWC'。
    :param weights: 不同预测时间步的评价指标占比。如果为 None，则表示所有预测时间步的评价指标占比一致。仅支持 list。长度与预测时间步数相同。
    :return:
    """
    writer = Writer(save_dir, is_delete=True)  # 创建结果保存目录
    # 检查结果目录 .identify 文件
    check_result_identify(lower_dir)
    check_result_identify(upper_dir)
    start_time = time.perf_counter()  # 记录开始时间
    print("目录检查通过，开始读取预测结果...")
    writer.add_text("目录检查通过，开始读取预测结果...", filename="Logs", folder="documents", suffix="log")

    # 读取 lower_dir 和 upper_dir 中的预测结果（key 是变量名，value 是预测结果（包含多个模型和多步预测））
    variable_set, model_set = set(), set()  # 用于保存 lower_fold 和 upper_fold 中的共有的变量名和模型名
    forecast_steps, columns_names = 0, []  # 用于保存预测步数和列名（下界和上界交集的列名）
    actual_result = [dict(), dict(), dict(), dict()]  # actual_fold_result, actual_train_result, actual_valid_result, actual_test_result
    lower_result = [dict(), dict(), dict(), dict()]  # lower_fold_result, lower_train_result, lower_valid_result, lower_test_result
    upper_result = [dict(), dict(), dict(), dict()]  # upper_fold_result, upper_train_result, upper_valid_result, upper_test_result
    pattern = [re.compile('(?<=^fold predict \()\S+(?=\).xlsx)'), re.compile('(?<=^train predict \()\S+(?=\).xlsx)'),
               re.compile('(?<=^valid predict \()\S+(?=\).xlsx)'), re.compile('(?<=^test predict \()\S+(?=\).xlsx)')]
    # 读取结果
    lower_dir, upper_dir = os.path.join(lower_dir, 'results'), os.path.join(upper_dir, 'results')
    # 执行读取 lower_dir 中的预测结果
    for file in os.listdir(lower_dir):
        for idx in range(len(pattern)):
            if re.search(pattern[idx], file):
                variable_name = re.search(pattern[idx], file).group()  # 获取变量名
                # 读取指定预测结果并保存
                lower_result[idx][variable_name] = pd.read_excel(os.path.join(lower_dir, file), index_col=0, header=0).reset_index(drop=True)
                # 获取并更新模型名（交集）
                temp_model_set = set(
                    [re.search(r"\S+(?=_-\d+)", col).group() for col in lower_result[idx][variable_name].columns]
                ).difference([variable_name])  # 获取模型名集合
                model_set = model_set & temp_model_set if model_set else temp_model_set  # 更新模型名集合
    # 执行验证 lower_dir 读取是否完整
    assert all([lower_result[i].keys() == lower_result[i + 1].keys() for i in range(len(lower_result) - 1)]), \
        "lower_dir 中的预测变量结果读取不完整！"
    variable_set = set(lower_result[0].keys())  # 获取 lower_dir 中的变量名集合
    # 执行读取 upper_dir 中的预测结果
    for file in os.listdir(upper_dir):
        for idx in range(len(pattern)):
            if re.search(pattern[idx], file):
                variable_name = re.search(pattern[idx], file).group()  # 获取变量名
                # 读取指定预测结果并保存
                upper_result[idx][variable_name] = pd.read_excel(os.path.join(upper_dir, file), index_col=0, header=0).reset_index(drop=True)
                # 获取并更新模型名（交集）
                temp_model_set = set(
                    [re.search(r"\S+(?=_-\d+)", col).group() for col in upper_result[idx][variable_name].columns]
                ).difference([variable_name])  # 获取模型名集合
                model_set = model_set & temp_model_set if model_set else temp_model_set  # 更新模型名集合
    # 执行验证 upper_dir 读取是否完整
    assert all([upper_result[i].keys() == upper_result[i + 1].keys() for i in range(len(upper_result) - 1)]), \
        "upper_dir 中的预测变量结果读取不完整！"
    variable_set = variable_set & set(upper_result[0].keys())  # 获取 lower_dir 和 upper_dir 变量名集合的交集
    print("预测结果读取完成，开始分离真实值和预测值...")
    writer.add_text("预测结果读取完成，开始分离真实值和预测值...", filename="Logs", folder="documents", suffix="log")

    # 分离真实值和预测值（包含验证）
    assert len(lower_result) == len(upper_result), "lower_dir 和 upper_dir 中的预测数据集数量不一致，或者缺少部分数据集！"  # 错误一般来源于数据结果不完整
    for idx in range(len(lower_result)):
        for var in variable_set:  # 只选取共有变量
            # 获取实际值
            lower_actual = lower_result[idx][var].filter(regex=f"^{var}_-\d+")
            upper_actual = upper_result[idx][var].filter(regex=f"^{var}_-\d+")
            assert lower_actual.equals(upper_actual), f"lower_dir 和 upper_dir 中的 {var} 实际值不一致！"
            forecast_steps = lower_actual.shape[1] if forecast_steps == 0 else forecast_steps  # 获取预测步数
            assert forecast_steps == lower_actual.shape[1], "lower_dir 和 upper_dir 中的预测步数不一致！"
            columns_names = [f"{model}_-{i}" for model in model_set for i in range(1, forecast_steps+1)]  # 获取预测结果的列名
            actual_result[idx][var] = lower_actual[lower_actual.columns.tolist() * (len(columns_names) // forecast_steps)]  # 保存实际值（通过复制保持与实际上界下界尺寸相同）
            lower_result[idx][var] = lower_result[idx][var][columns_names]  # 更新下界预测结果
            upper_result[idx][var] = upper_result[idx][var][columns_names]  # 更新上界预测结果
    print("实际值和预测值分离完成，开始计算概率预测评价指标...")
    writer.add_text("实际值和预测值分离完成，开始计算概率预测评价指标...", filename="Logs", folder="documents", suffix="log")

    # 处理 weights 参数
    if weights:
        assert isinstance(weights, list) and len(weights) == forecast_steps, \
            f"weights 必须为长度为 {forecast_steps} 的 list，当前 weights 为 {weights}！"
    else:
        weights = [1 / forecast_steps] * forecast_steps  # 默认每个预测时间步的权重相同
    writer.add_text(f"预测步数：{forecast_steps}；预测模型：{'、'.join(model_set)}；多步预测权重：{weights}",
                    filename="Logs", folder="documents", suffix="log")

    # 计算概率预测评价指标
    metrics_sub = [[], [], [], []]  # 每个数据集的每列评价指标
    metrics_all = [[], [], [], []]  # 每个数据集的总体评价指标
    for idx in range(len(lower_result)):
        for var in variable_set:
            lower_value = lower_result[idx][var]  # 下界预测结果
            upper_value = upper_result[idx][var]  # 上界预测结果
            actual_value = actual_result[idx][var]  # 实际值
            assert lower_value.columns.equals(upper_value.columns), "下界和上界预测结果的列名不一致！"
            assert [col.split('_-')[-1] for col in lower_value.columns] == [col.split('_-')[-1] for col in actual_value.columns], \
                "预测值和实际值对应的时间步不一致！"
            actual_value.columns = lower_value.columns  # 保证实际值的列名与下界和上界预测结果一致
            # 计算每个数据集的每列评价指标
            for col in lower_value.columns:
                metrics_dict = calculate_possibility(
                    actual_value=actual_value[[col]],
                    lower_value=lower_value[[col]],
                    upper_value=upper_value[[col]],
                    alpha=alpha, metrics=metrics
                )
                model_name, target_step = col.split('_-')
                metrics_sub[idx].append({'target': var, 'model': model_name, 'step': target_step, **metrics_dict})
            # 计算每个数据集的总体评价指标
            for model in model_set:
                metrics_dict = calculate_possibility(
                    actual_value=actual_value.filter(regex=f"^{model}_-\d+"),
                    lower_value=lower_value.filter(regex=f"^{model}_-\d+"),
                    upper_value=upper_value.filter(regex=f"^{model}_-\d+"),
                    alpha=alpha, metrics=metrics, weights=weights
                )
                metrics_all[idx].append({'target': var, 'model': model, **metrics_dict})
    fold_metrics_sub, train_metrics_sub, valid_metrics_sub, test_metrics_sub = \
        [pd.DataFrame(m, index=range(1, len(m)+1)) for m in metrics_sub]
    fold_metrics_all, train_metrics_all, valid_metrics_all, test_metrics_all = \
        [pd.DataFrame(m, index=range(1, len(m)+1)) for m in metrics_all]
    print("概率预测评价指标计算完成，开始保存结果...")
    writer.add_text("概率预测评价指标计算完成，开始保存结果...", filename="Logs", folder="documents", suffix="log")

    # 保存概率预测评价指标结果
    writer.add_df(fold_metrics_all, axis=0, filename="fold metrics", folder="results", suffix="xlsx",
                  reset_index=True, reset_drop=True, sort_list=['target', 'model'], sort_ascending=True)
    writer.add_df(train_metrics_all, axis=0, filename="train metrics", folder="results", suffix="xlsx",
                  reset_index=True, reset_drop=True, sort_list=['target', 'model'], sort_ascending=True)
    writer.add_df(valid_metrics_all, axis=0, filename="valid metrics", folder="results", suffix="xlsx",
                  reset_index=True, reset_drop=True, sort_list=['target', 'model'], sort_ascending=True)
    writer.add_df(test_metrics_all, axis=0, filename="test metrics", folder="results", suffix="xlsx",
                  reset_index=True, reset_drop=True, sort_list=['target', 'model'], sort_ascending=True)
    writer.add_df(fold_metrics_sub, axis=0, filename="fold metrics (detailed)", folder="results", suffix="xlsx",
                  reset_index=True, reset_drop=True, sort_list=['target', 'model', 'step'], sort_ascending=True)
    writer.add_df(train_metrics_sub, axis=0, filename="train metrics (detailed)", folder="results", suffix="xlsx",
                  reset_index=True, reset_drop=True, sort_list=['target', 'model', 'step'], sort_ascending=True)
    writer.add_df(valid_metrics_sub, axis=0, filename="valid metrics (detailed)", folder="results", suffix="xlsx",
                  reset_index=True, reset_drop=True, sort_list=['target', 'model', 'step'], sort_ascending=True)
    writer.add_df(test_metrics_sub, axis=0, filename="test metrics (detailed)", folder="results", suffix="xlsx",
                  reset_index=True, reset_drop=True, sort_list=['target', 'model', 'step'], sort_ascending=True)
    writer.write(save_mode='a+')  # 保存结果
    use_time = time.perf_counter() - start_time  # 计算运行时间
    print(f"概率预测评价指标结果保存完成！运行时间为：{use_time} 秒。")
    writer.add_text(f"概率预测评价指标结果保存完成！运行时间为：{use_time} 秒。",
                    filename="Logs", folder="documents", suffix="log", save_mode='a+')


if __name__ == "__main__":
    alpha1 = 0.10  # 分位数，表示预测区间的置信水平。例如 0.05 表示 95% 的置信水平
    lower_dir1 = r"E:\Program\Pycharm\Predictor\results\Ensemble(Compare, alpha=0.10)"
    upper_dir1 = r"E:\Program\Pycharm\Predictor\results\Ensemble(Compare, alpha=0.90)"
    save_dir1 = r"E:\Program\Pycharm\Predictor\results\Probabilistic(Compare, alpha=0.10-0.90)"
    calculate_possibility_folds(lower_dir1, upper_dir1, save_dir1, alpha=alpha1, metrics=None, weights=None)
