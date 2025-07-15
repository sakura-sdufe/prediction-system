# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/30 16:57
# @Author   : 张浩
# @FileName : read.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import re
import os
import yaml
import torch
import pickle
import warnings
import traceback
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, Dataset, DataLoader
from datetime import datetime
from typing import List, Tuple, Union, Literal

from .selector import Selector


def read_yaml(file_path):
    """
    读取 yaml 文件，并将其转换为字典格式。
    :param file_path: yaml 文件路径
    :return: 字典格式的数据
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data


def read_pickle(filepath):
    """
    使用 pickle 模块以二进制的形式读取文件。
    :param filepath: 读取的文件路径
    :return: None
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(data, file_path):
    """
    使用 pickle 模块以二进制的形式保存文件。
    :param data: 需要保存的内容。
    :param file_path: 保存的文件路径。
    :return: None
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def write_result_identify(result_dir, version):
    """
    将结果目录的识别文件写入到指定目录。
    :param result_dir: 结果目录
    :param version: 版本号
    :return: None
    """
    assert os.path.isdir(result_dir), f"{result_dir} 不是一个目录或该目录不存在。"
    with open(os.path.join(result_dir, ".identify"), 'wb') as f:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        IDENTIFY = f"VERSION: {version}\nTIME: {current_time}\nPATH: {result_dir}"
        f.write(IDENTIFY.encode('utf-8'))


def check_result_identify(result_dir):
    """
    检查结果目录是否存在，并读取 .identify 识别文件。
    :param result_dir:
    :return: IDENTIFY_VERSION, IDENTIFY_TIME, IDENTIFY_PATH 分别表示 .identify 识别文件中的版本号、时间和路径。
    """
    assert os.path.isdir(result_dir), f"{result_dir} 不是一个目录或该目录不存在。"
    # 读取 .identify 识别文件
    assert os.path.exists(os.path.join(result_dir, ".identify")), "未找到 .identify 识别文件，无法读取预测结果和特征值。"
    IDENTIFY_KEYS = {'VERSION', 'TIME', 'PATH'}
    IDENTIFY_VERSION, IDENTIFY_TIME, IDENTIFY_PATH = None, None, None  # 初始化变量
    with open(os.path.join(result_dir, ".identify"), 'rb') as f:  # 读取识别文件
        IDENTIFY_read = f.read().decode('utf-8')
    IDENTIFY_lines = IDENTIFY_read.split('\n')
    for line in IDENTIFY_lines:
        key, value = line.split(': ')
        if key == 'PATH':
            assert os.path.samefile(value, result_dir), "识别文件中的路径与当前路径不一致。"
            IDENTIFY_PATH = value
            IDENTIFY_KEYS.remove(key)
        elif key == 'VERSION':
            IDENTIFY_VERSION = float(value)
            IDENTIFY_KEYS.remove(key)
        elif key == 'TIME':
            IDENTIFY_TIME = value
            IDENTIFY_KEYS.remove(key)
    assert not IDENTIFY_KEYS, "识别文件内容不完整。"  # 检查 IDENTIFY_VERSION, IDENTIFY_TIME, IDENTIFY_PATH 是否都存在
    return IDENTIFY_VERSION, IDENTIFY_TIME, IDENTIFY_PATH


def split_sample(feature, target, start_position, end_position):
    """
    将 DataFrame 或者 array 类型数据进行切分（按照 dim=0 维度切分），切分的位置由 start_position 和 end_position 控制。
    :param feature: 特征数据。数据类型为 np.ndarray、pd.DataFrame 或者 torch.tensor。
    :param target: 目标数据。数据类型为 np.ndarray、pd.DataFrame 或者 torch.tensor。
    :param start_position: 切分的起始位置。如果是介于 0 和 1 之间的小数，则表示按照比例切分。如果是大于 1 的整数，则表示按照数量切分。
    :param end_position: 切分的结束位置。如果是介于 0 和 1 之间的小数，则表示按照比例切分。如果是大于 1 的整数，则表示按照数量切分（不包含 end_position）。
        Note: predict 和 targets 也可以接受其他可以被 len 函数调用和可以切片的数据类型。
            当 start_position 和 end_position 取值为 0.0 和 1.0 时，表示按照比例切分（输入为 float 类型数据按比例切分）；
            当 start_position 和 end_position 取值为 0 和 1 时，表示按照数量切分（输入为 int 类型数据按数量切分）。
    :return: feature_split, target_split 分别表示指定范围内的特征数据和目标数据。
    """
    assert type(start_position) == type(end_position), "start_position 和 end_position 的数据类型必须相同！"
    assert (start_position>=0) and (end_position>=0), "start_position 和 end_position 的取值必须大于等于 0！"
    assert start_position <= end_position, "start_position 的取值不能大于 end_position！"
    if len(feature) != len(target):
        warnings.warn(
            "start_position 和 end_position 在 dim=0 维度上长度不相同；如果你使用的时按比例划分，那么使用 predict 的长度。",
            category=UserWarning)

    if isinstance(start_position, float):  # 按照比例切分
        if (start_position<=1) and (end_position<=1):
            start_position = round(start_position * len(feature))
            end_position = round(end_position * len(feature))
        else:
            raise ValueError("当 start_position 和 end_position 按照比例划分时，取值必须在 0 和 1 之间！")

    assert end_position <= len(feature), "end_position 的取值不能大于特征数据的长度！"
    return feature[start_position:end_position], target[start_position:end_position]


def _add_history_feature(time_feature: pd.DataFrame, time_target: pd.DataFrame, time_unknown_variables, time_known_variables,
                         time_step, output_size, is_features_history=False):
    """
    将 DataFrame 中的数据按照时变已知变量、时变未知变量、目标变量分别处理，返回的特征可以用于回归任务。
    :param time_feature: 特征数据（其他影响预测数据的变量），接受 DataFrame 类型数据。
    :param time_target: 目标数据（需要预测的数据），接受 DataFrame 类型数据。
    :param time_unknown_variables: 时变未知变量，接受 list 和 tuple 类型（内部的每个元素是字符串，表示 DataFrame 的列名）。
    :param time_known_variables: 时变已知变量，接受 list 和 tuple 类型（内部的每个元素是字符串，表示 DataFrame 的列名）。
    :param time_step: 使用前多少步的数据进行预测。
    :param output_size: 目标输出的个数。
    :param is_features_history: 是否使用 predict 中的历史数据作为特征，默认为 False。
        如果使用特征历史数据，那么将会根据历史时间步为每个时变未知变量、时变已知变量、目标变量分别添加历史数据
        （其中时变未知变量和时变已知变量均会创建 num_predictors 个列）；
        如果不使用特征历史数据，那么只会添加最新观测的数据（时变已知变量用当前时间步表示，时变未知变量用上一时间步表示）。
    :return: 返回处理后的时间序列特征数据和目标数据，均为 DataFrame 类型数据。
    Note:
        1. 如果 time_feature 中存在 time_unknown_variables 和 time_known_variables 未包含的变量，那么那些变量将不会加入到返回的时间序列特征中。
        2. 返回的时间序列特征将会对列重新命名，列的命名格式为：'变量名_i' 表示前 i 步的数据，'变量名_-i' 表示未来 i 步数据
            （这只会存在于时变已知变量和目标中）。
        3. 如果 add_history_features 为 True，返回的特征的尺度为 [样本数，(输入特征数+预测目标个数)*时间步]，
            如果 add_history_features 为 False，返回的特征的尺度为 [样本数，(输入特征数+预测目标个数)*预测步数]，
            目标的尺度为 [样本数，预测步数*预测目标个数]。
        4. feature 特征排列顺序为：时变已知、时变未知、目标。
        5. 该函数将会返回 DataFrame 类型数据，一般适用于回归任务，并不适合循环神经网络模型。
    """
    assert time_feature.shape[0] == time_target.shape[0], "特征数据和目标数据的长度不一致！"
    # Step 1.1：创建一个新的 DataFrame 用于存储时间序列特征数据
    feature = pd.DataFrame()

    # Step 1.2：如果变量在 time_unknown_variables 和 time_known_variables 中存在，但 time_feature 中不存在情况，抛出错误。
    for col in time_unknown_variables:
        if col not in time_feature.columns:
            raise ValueError(f"时变未知变量 '{col}' 不在数据集中，无法进行转换！")
    for col in time_known_variables:
        if col not in time_feature.columns:
            raise ValueError(f"时变已知变量 '{col}' 不在数据集中，无法进行转换！")
    time_unknown_variables = list(time_unknown_variables)
    time_known_variables = list(time_known_variables)

    # 计算需要添加的历史步
    if is_features_history:  # 如果使用特征历史数据，那么需要添加 时间步 个历史信息；
        history_step = time_step
    else:  # 如果不使用特征历史数据，那么只需要添加 预测步 个历史信息；
        history_step = output_size

    # Step 2：处理时变已知变量（需要判断是否加入历史数据）
    feature_known = pd.DataFrame()  # 用于存储时变已知变量的历史数据
    time_known_feature = time_feature[time_known_variables].iloc[output_size:, :]
    for i in range(history_step):
        feature_shift = time_known_feature.shift(i)
        if i >= output_size:
            feature_shift.columns = [f"{col}_{i - output_size + 1}" for col in feature_shift.columns]
        elif i < output_size:
            feature_shift.columns = [f"{col}_{i - output_size}" for col in feature_shift.columns]
        feature_known = pd.concat([feature_known, feature_shift], axis=1)
    feature_known = feature_known[time_step-1:]  # 去掉前 num_predictors 行（对齐）
    feature_known.reset_index(drop=True, inplace=True)
    feature = pd.concat([feature, feature_known], axis=1)

    # Step 3：处理时变未知变量（需要判断是否加入历史数据）
    feature_unknown = pd.DataFrame()  # 用于存储时变未知变量的历史数据
    time_unknown_feature = time_feature[time_unknown_variables].iloc[:time_feature.shape[0] - output_size + 1, :]
    for i in range(1, history_step+1):
        feature_shift = time_unknown_feature.shift(i)
        feature_shift.columns = [f"{col}_{i}" for col in feature_shift.columns]
        feature_unknown = pd.concat([feature_unknown, feature_shift], axis=1)

    feature_unknown = feature_unknown[time_step:]  # 去掉前 num_predictors 行（对齐）
    feature_unknown.reset_index(drop=True, inplace=True)
    feature = pd.concat([feature, feature_unknown], axis=1)

    # Step 4：处理目标变量，加入目标变量的历史数据
    for i in range(1, time_step+1):
        target_shift = time_target.iloc[:time_target.shape[0] - output_size + 1, :].shift(i)
        target_shift.columns = [f"{col}_{i}" for col in time_target.columns]
        target_shift = target_shift.iloc[time_step:, :]  # 去掉前 num_predictors 行（这些行包括 NaN 值）
        target_shift.reset_index(drop=True, inplace=True)
        feature = pd.concat([feature, target_shift], axis=1)

    # Step 5：处理目标（多输出）
    target = pd.concat(
        [time_target.iloc[time_step:, :].shift(i) for i in range(output_size - 1, -1, -1)], axis=1
    )
    target = target.iloc[output_size-1:, :]  # 多输出对齐
    target.reset_index(drop=True, inplace=True)
    target.columns = [f"{col}_-{i}" for i in range(1, output_size+1) for col in time_target.columns]

    assert len(feature) == len(target), "特征数据和目标数据的长度不一致。【请检查该函数内部】"
    return feature, target


def _add_time_feature(time_feature: pd.DataFrame, time_target: pd.DataFrame, time_unknown_variables, time_known_variables,
                      time_step, output_size, dtype=torch.float32):
    """
    将 DataFrame 中的数据按照时变已知变量、时变未知变量、目标变量分别处理，返回的特征可以用于时序任务。
    :param time_feature: 特征数据（其他影响预测数据的变量），接受 DataFrame 类型数据。
    :param time_target: 目标数据（需要预测的数据），接受 DataFrame 类型数据。
    :param time_unknown_variables: 时变未知变量，接受 list 和 tuple 类型（内部的每个元素是字符串，表示 DataFrame 的列名）。
    :param time_known_variables: 时变已知变量，接受 list 和 tuple 类型（内部的每个元素是字符串，表示 DataFrame 的列名）。
    :param time_step: 使用前多少步的数据进行预测。
    :param output_size: 目标输出的个数。
    :param dtype: 转换后的数据类型，默认为 torch.float32。
    :return: 返回处理后的时间序列特征数据和目标数据，均为 torch.tensor 类型数据；并返回特征列对应的列名（feature_variables），长度=输入特征数+预测目标个数。
    Note:
        1. 如果 feature 中存在 time_unknown_variables 和 time_known_variables 未包含的变量，那么那些变量将不会加入到返回的时间序列特征中。
        2. feature 特征排列顺序为：时变已知、时变未知、目标。
        2. 返回的特征的尺度为 [样本数，时间步，(输入特征数+预测目标个数)]，目标的尺度为 [样本数，预测步数*预测目标个数]。
        3. 该函数将会返回 torch.tensor 类型数据，一般适用于时序预测任务，并不适合回归任务。
    """
    assert time_feature.shape[0] == time_target.shape[0], "特征数据和目标数据的长度不一致！"
    sample_number = time_target.shape[0] - output_size - time_step + 1

    # Step 1：如果变量在 time_unknown_variables 和 time_known_variables 中存在，但 time_feature 中不存在情况，抛出错误。
    for col in time_unknown_variables:
        if col not in time_feature.columns:
            raise ValueError(f"时变未知变量 '{col}' 不在数据集中，无法进行转换！")
    for col in time_known_variables:
        if col not in time_feature.columns:
            raise ValueError(f"时变已知变量 '{col}' 不在数据集中，无法进行转换！")
    time_unknown_variables = list(time_unknown_variables)
    time_known_variables = list(time_known_variables)

    # Step 2.1：处理特征（提取时变未知和时变已知特征）
    feature_unknown = torch.tensor(pd.concat([time_feature[time_unknown_variables], time_target], axis=1).values, dtype=dtype)
    feature_known = torch.tensor(time_feature[time_known_variables].values, dtype=dtype)
    target = torch.tensor(time_target.values, dtype=dtype)

    # Step 2.2：处理特征（特征对齐）
    feature_unknown_shift = torch.stack(
        [feature_unknown[ind: ind + time_step, :] for ind in list(range(sample_number))]
    )
    feature_known_shift = torch.stack(
        [feature_known[ind + output_size: ind + output_size + time_step, :] for ind in list(range(sample_number))]
    )
    target_shift = torch.stack(
        [target[ind + time_step: ind + time_step + output_size, :] for ind in list(range(sample_number))]
    )
    feature = torch.concat([feature_known_shift, feature_unknown_shift], dim=-1)  # 尺寸为 [样本数，时间步，输入特征数]
    target = target_shift.reshape(target_shift.shape[0], -1)  # 将目标数据 reshape 成 [样本数，预测步数*预测目标个数]
    assert feature.shape[0] == target.shape[0], "转换出现未预料的错误：特征数据和目标数据的长度不一致！"
    assert (feature.ndim == 3) and (target.ndim == 2), "转换出现未预料的错误：特征或目标数据维度不匹配！"
    return feature, target


class ReadDataset:
    def __init__(self, path: str, time_step: int, output_size: int, writer, add_history_features: bool=True,
                 normalization: type=None):
        """
        处理数据集文件，并将其转换为机器学习或深度学习数据格式。
        :param path: 文件路径。
        :param time_step: 时间步长。
        :param output_size: 预测时间步长。
        :param writer: 实例化后的 Writer 对象。
        :param add_history_features: 是否添加历史特征（包括时变已知、时变未知特征），该参数仅对回归任务有效，默认为 True。
        :param normalization: 数据归一化方法，需要包含 get_norm_result, norm, denorm 方法。如果需要使用归一化方法，那么需要传入一个归一化对象。
        Note:
            1. 输入特征数 = 时变已知特征数 + 时变未知特征数；样本数 = (数据长度 - 预测步数 - 时间步 + 1)  // 采样间隔。
            2. 如果 add_history_features=True，那么返回的特征和目标值中会包含历史特征和目标值，此时特征的尺寸为 [样本数，(输入特征数+预测目标个数)*时间步]。
            3. 如果 add_history_features=False，那么返回的特征和目标值中不会包含历史特征和目标值，此时特征的尺寸为 [样本数，输入特征数+预测目标个数*时间步]；
                此时时序未知特征仅包含上一个时间步的信息，时变已知特征仅包含当前时间步的信息，历史目标值包含前 time_step 个时间步的信息。
            4. 目标的尺寸为 [样本数，预测目标个数*预测步数]。
        """
        self.file_path, self.time_step, self.output_size = path, time_step, output_size
        self.add_history_features, self.normalization = add_history_features, normalization
        self.input_size_reg, self.input_size_seq = None, None  # 初始化回归任务和时序任务的输入尺寸
        self.writer = writer

        # self.get_dataset（数据集读取）、self.get_feature（特征获取）、self.get_target（目标获取）
        self.feature_variables, self.target_variables = None, None  # 初始化特征和目标变量，数据类型均为 tuple。
        self.feature_reg_columns, self.feature_seq_columns = None, None  # 初始化回归任务和时序任务的特征列，数据类型均为 tuple。
        self.target_columns = None  # 初始化目标列（长度为目标变量个数乘以预测步数，等于预测模型输出个数），数据类型为 tuple。
        self.data, self.time_feature, self.time_target = None, None, None  # 初始化数据集、特征和目标值，数据类型均为 pd.DataFrame。

        # self.select_feature（特征选择）
        self.selector, self.selected_feature = None, None  # 初始化特征选择器和选择后的特征，数据类型分别为 Selector 和 pd.DataFrame。
        self.time_known_variables, self.time_unknown_variables = None, None  # 初始化时变已知和时变未知变量（随特征选择更新），数据类型均为 tuple。
        self.selected_feature_variables = None  # 初始化选择后的特征变量（未执行特征选择则等于 self.feature_variables），数据类型为 tuple。

        # 数据归一化（对所有特征进行归一化操作，即对 self.time_feature 进行归一化操作）
        if self.normalization:
            self.normalization_feature, self.normalization_target = None, None  # 初始化归一化方法，数据类型均为 self.normalization。
            self.time_feature_norm, self.time_target_norm = None, None  # 初始化归一化后的特征和目标值，数据类型均为 pd.DataFrame。

        # 回归数据集封装，使用选择后的特征进行封装。
        self.feature_reg, self.target_reg = None, None  # 初始化特征和目标值（添加历史特征），数据类型均为 pd.DataFrame。
        if self.normalization:
            self.feature_reg_norm, self.target_reg_norm = None, None  # 初始化标准化特征和目标值（添加历史特征），数据类型均为 pd.DataFrame。

        # 时序数据集封装，使用选择后的特征进行封装。
        self.feature_seq, self.target_seq = None, None  # 初始化时序特征和目标值，数据类型均为 3D torch.tensor，尺寸为 [样本数，时间步，输入特征]
        if self.normalization:
            self.feature_seq_norm, self.target_seq_norm = None, None

        self.read_dataset()  # 读取数据集

    def read_dataset(self, **kwargs) -> pd.DataFrame:
        """
        读取数据集文件（仅支持 xlsx xls csv 文件），对 self.data 进行赋值（如果需要自定义可以手动赋值）。
        :param kwargs: 读取文件的其他参数。
        :return: 读取的数据集（pd.DataFrame）。
        """
        if self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            self.data = pd.read_excel(self.file_path, **kwargs)
        elif self.file_path.endswith('.csv'):
            self.data = pd.read_csv(self.file_path, **kwargs)
        else:
            raise ValueError("文件格式不支持！")
        # 尝试将非数值型数据转换为数值型，如果转换失败则保留原数据
        for col in self.data.select_dtypes(include=['object']).columns:
            try:
                self.data[col] = pd.to_numeric(self.data[col])
            except ValueError:
                print(f"列 '{col}' 无法转换为数值型数据，保留原数据！")
                self.writer.add_text(f"列 '{col}' 无法转换为数值型数据，保留原数据！",
                                     filename="Logs", folder="documents", suffix="log", save_mode='a+')
        return self.data

    def set_feature(self, time_known: Union[List[str], Tuple[str]], time_unknown: Union[List[str], Tuple[str]]):
        """
        获取特征值，并对 self.time_feature 进行赋值（如果需要自定义可以手动赋值）。
        :param time_known: 时变已知变量列表或元组。
        :param time_unknown: 时变未知变量列表或元组。
        :return: None
        """
        assert isinstance(time_known, (list, tuple)), "time_known 必须是一个列表或元组。"
        assert all(isinstance(col, str) for col in time_known), "time_known 列表中的元素必须是字符串。"
        assert isinstance(time_unknown, (list, tuple)), "time_unknown 必须是一个列表或元组。"
        assert all(isinstance(col, str) for col in time_unknown), "time_unknown 列表中的元素必须是字符串。"
        self.time_known_variables, self.time_unknown_variables = [], []
        for col in time_known:  # 遍历时变已知变量列表，筛除不在数据集中的变量
            if col not in self.data.columns:
                warnings.warn(f"时变已知变量 '{col}' 不在数据集中，已忽略！")
            else:
                self.time_known_variables.append(col)
        for col in time_unknown:  # 遍历时变未知变量列表，筛除不在数据集中的变量
            if col not in self.data.columns:
                warnings.warn(f"时变未知变量 '{col}' 不在数据集中，已忽略！")
            else:
                self.time_unknown_variables.append(col)
        self.feature_variables = self.time_known_variables + self.time_unknown_variables
        self.time_feature = self.data[self.feature_variables]  # 选择特征值 pd.DataFrame
        self.selected_feature = self.time_feature  # 初始化选择后的特征
        # 将变量名列表转换为元组
        self.time_known_variables = tuple(self.time_known_variables)
        self.time_unknown_variables = tuple(self.time_unknown_variables)
        self.feature_variables = tuple(self.feature_variables)
        self.selected_feature_variables = self.feature_variables  # 初始化选择后的特征变量

    def set_target(self, target: Union[List[str], Tuple[str]]):
        """
        获取目标值，并对 self.time_target 进行赋值（如果需要自定义可以手动赋值）。
        :param target: 目标值的列名列表或元组。
        :return: None
        """
        assert isinstance(target, (list, tuple)), "columns 必须是一个列表或元组。"
        assert all(isinstance(col, str) for col in target), "columns 列表中的元素必须是字符串。"
        self.target_variables = []
        for col in target:  # 遍历目标值列表，筛除不在数据集中的变量
            if col not in self.data.columns:
                warnings.warn(f"目标值 '{col}' 不在数据集中，已忽略！")
            else:
                self.target_variables.append(col)
        self.time_target = self.data[self.target_variables]  # 选择目标值 pd.DataFrame
        self.target_variables = tuple(self.target_variables)  # 将变量名列表转换为元组

    def select_feature(self, method, number, how: Literal['inner', 'outer'] = 'outer'):
        """
        执行特征选择，并对 self.selected_feature 进行赋值（如果需要自定义可以手动赋值）。
        :param method: 特征选择方法。可选值为 None、'互信息'、'F检验'、'卡方检验'、'相关系数'（统计学筛选）；
                        以及 'sMAPE'、'MAPE'、'MSE'、'RMSE'、'MAE'、'R2'（指标筛选）。
                        不建议特征选择使用指标筛选方法，因为指标筛选是通过计算不同特征列与目标列的评价指标值进行筛选，所以指标筛选方法更适合模型集成使用。
        :param number: 每个目标列特征选择后特征的数量。由于不同目标列选择出来的结果可能不同，所以最终筛选出来的特征数量可能会不等于 number。
        :param how: 特征选择的合并方式。可选值为 'inner' 和 'outer'，分别表示交集和并集。
        :return: None
        """
        print(f"正在使用 {method} 方法选择 {number} 个特征，并且使用 {how} 方式进行合并...")
        self.writer.add_text(f"使用 {method} 方法选择 {number} 个特征，并且使用 {how} 方式进行合并。",
                             filename="Logs", folder="documents", suffix="log", save_mode='a+')
        self.selector = Selector(method=method, number=number)
        self.selected_feature_variables = []  # 重置选择后的特征变量
        for col in self.time_target.columns:
            temp_selected = self.selector.fit_transform(self.time_feature, self.time_target[col])  # temp_selected 是 DataFrame 类型（包含列名）
            if temp_selected.shape[1] == 0:
                warnings.warn(f"目标值 '{col}' 没有选择到特征，已忽略！")
            else:
                temp_selected_columns = temp_selected.columns.tolist()
                if not self.selected_feature_variables:
                    self.selected_feature_variables = temp_selected_columns
                else:
                    if how == 'inner':
                        self.selected_feature_variables = list(set(self.selected_feature_variables) & set(temp_selected_columns))
                    elif how == 'outer':
                        self.selected_feature_variables = list(set(self.selected_feature_variables) | set(temp_selected_columns))
                    else:
                        raise ValueError("how 参数错误！")
        self.selected_feature = self.time_feature[self.selected_feature_variables]  # 选择特征值 pd.DataFrame
        self.selected_feature_variables = tuple(self.selected_feature_variables)  # 将变量名列表转换为元组
        # 更新 self.time_known_variables 和 self.time_unknown_variables
        self.time_known_variables = tuple([col for col in self.selected_feature_variables if col in self.time_known_variables])
        self.time_unknown_variables = tuple([col for col in self.selected_feature_variables if col in self.time_unknown_variables])

    def convert(self):
        """为传入的时间序列特征和目标添加历史信息，用于回归数据集的封装。"""
        # 回归数据集封装（添加历史特征）
        self.feature_reg, self.target_reg = _add_history_feature(
            time_feature=self.selected_feature, time_target=self.time_target,
            time_unknown_variables=list(self.time_unknown_variables),
            time_known_variables=list(self.time_known_variables),
            time_step=self.time_step, output_size=self.output_size,
            is_features_history=self.add_history_features
        )
        # 时序数据集封装（添加时间特征）
        self.feature_seq, self.target_seq = _add_time_feature(
            time_feature=self.selected_feature, time_target=self.time_target,
            time_unknown_variables=list(self.time_unknown_variables),
            time_known_variables=list(self.time_known_variables),
            time_step=self.time_step, output_size=self.output_size,
        )
        if self.normalization:
            # 对特征和目标分别执行标准化操作
            self.normalization_feature = self.normalization(self.time_feature)
            self.normalization_target = self.normalization(self.time_target)
            self.time_feature_norm = self.normalization_feature.get_norm_result()
            self.time_target_norm = self.normalization_target.get_norm_result()
            # 回归数据集封装（添加历史特征）
            self.feature_reg_norm, self.target_reg_norm = _add_history_feature(
                time_feature=self.time_feature_norm[list(self.selected_feature_variables)],
                time_target=self.time_target_norm,
                time_unknown_variables=list(self.time_unknown_variables),
                time_known_variables=list(self.time_known_variables),
                time_step=self.time_step, output_size=self.output_size,
                is_features_history=self.add_history_features
            )
            # 时序数据集封装（添加时间特征）
            self.feature_seq_norm, self.target_seq_norm = _add_time_feature(
                time_feature=self.time_feature_norm[list(self.selected_feature_variables)],
                time_target=self.time_target_norm,
                time_unknown_variables=list(self.time_unknown_variables),
                time_known_variables=list(self.time_known_variables),
                time_step=self.time_step, output_size=self.output_size,
            )
        # 更新参数
        self.feature_seq_columns = self.time_known_variables + self.time_unknown_variables + self.target_variables  # 时序特征的列标签
        self.feature_reg_columns = tuple(self.feature_reg.columns)  # 回归特征的列标签
        self.target_columns = tuple(self.target_reg.columns)  # 目标的列标签
        self.input_size_reg = self.feature_reg.shape[-1]  # 回归任务的输入特征数
        self.input_size_seq = self.selected_feature.shape[-1] + len(self.target_variables)  # 时序任务的输入特征数

    def get_regml(self, start: float = 0.0, end: float = 1.0, sample_gap: int=1, is_norm=True,
                  to_numpy: bool=False, fold=1, k=1, dtype=np.float32):
        """
        获取回归任务的机器学习数据集（用于 sklearn 模块）。
        :param start: 起始索引比例（浮点数）。
        :param end: 结束索引比例（浮点数）。
        :param is_norm: 返回的结果是否进行归一化处理。
        :param sample_gap: 采样间隔，默认为 1，表示连续采样；如果设置为大于 1 的值，表示每隔 sample_gap 个样本采样一个。
        :param to_numpy: 是否将结果转换为 numpy 数组，默认为 False。
        :param fold: 交叉验证的折数，默认为 1 表示不使用交叉验证，此时参数 k 不起作用。
        :param k: 交叉验证的第 k 折，值应当大于等于 1 小于等于 fold，默认为 1。
        :param dtype: 转换后的数据类型，默认为 np.float32。
        :return:
            split_feature: 分割后的特征，数据类型： pd.DataFrame 或 numpy.ndarray 或 Tuple。
            split_target: 分割后的目标，数据类型： pd.DataFrame 或 numpy.ndarray 或 Tuple。
        Note:
            1. 如果使用交叉验证，那么 split_feature 和 split_target 均为一个长度为 3 的元组，第一个值为用于训练的训练集，第二个值为用于评估的训练集，第三个值为验证集。
            2. 如果不使用交叉验证，那么只返回介于 start 和 end 之间的数据。
        """
        # 处理 k 折交叉验证
        k = 1 if fold == 1 else k
        fold_ratio = 1 / fold * (end - start)
        eval_start, eval_end = (k - 1) * fold_ratio + start, k * fold_ratio + start
        # 处理非标准化数据和标准化数据
        if is_norm and self.normalization:
            feature_reg, target_reg = self.feature_reg_norm, self.target_reg_norm
        elif not is_norm:
            feature_reg, target_reg = self.feature_reg, self.target_reg
        else:
            raise ValueError("未执行归一化操作，但是 is_norm=True")
        # 划分数据集
        train_feature_1, train_target_1 = split_sample(
            feature=feature_reg, target=target_reg, start_position=start, end_position=eval_start
        )
        train_feature_2, train_target_2 = split_sample(
            feature=feature_reg, target=target_reg, start_position=eval_end, end_position=end
        )
        eval_feature, eval_target = split_sample(
            feature=feature_reg, target=target_reg, start_position=eval_start, end_position=eval_end
        )
        train_feature = pd.concat([train_feature_1, train_feature_2], axis=0).reset_index(drop=True)
        train_target = pd.concat([train_target_1, train_target_2], axis=0).reset_index(drop=True)
        # 根据 train_feature 和 train_target 是否存在值判断是否使用交叉验证
        if train_feature.size == 0 and train_target.size == 0:  # 未使用交叉验证（训练特征和目标没有元素）
            # 下采样
            feature_sample = eval_feature.iloc[::sample_gap, :].reset_index(drop=True)
            target_sample = eval_target.iloc[::sample_gap, :].reset_index(drop=True)
            # 转换数据类型
            if to_numpy:
                feature_sample = feature_sample.to_numpy(dtype=dtype)
                target_sample = target_sample.to_numpy(dtype=dtype)
            else:
                feature_sample = feature_sample.astype(dtype)
                target_sample = target_sample.astype(dtype)
            return feature_sample, target_sample
        else:  # 使用交叉验证（训练特征和目标有元素）
            # 下采样
            train_feature_sample = train_feature.iloc[::sample_gap, :].reset_index(drop=True)
            train_target_sample = train_target.iloc[::sample_gap, :].reset_index(drop=True)
            # 转换数据类型
            if to_numpy:
                train_feature_sample = train_feature_sample.to_numpy(dtype=dtype)
                train_target_sample = train_target_sample.to_numpy(dtype=dtype)
                train_feature = train_feature.to_numpy(dtype=dtype)
                train_target = train_target.to_numpy(dtype=dtype)
                eval_feature = eval_feature.to_numpy(dtype=dtype)
                eval_target = eval_target.to_numpy(dtype=dtype)
            else:
                train_feature_sample = train_feature_sample.astype(dtype)
                train_target_sample = train_target_sample.astype(dtype)
                train_feature = train_feature.astype(dtype)
                train_target = train_target.astype(dtype)
                eval_feature = eval_feature.astype(dtype)
                eval_target = eval_target.astype(dtype)
            return (train_feature_sample, train_feature, eval_feature), (train_target_sample, train_target, eval_target)

    def get_regdl(self, batch_size: int, eval_batch_size: int=None, start: float = 0.0, end: float = 1.0,
                  sample_gap: int=1, shuffle: bool=False, is_norm: bool=True, fold=1, k=1, dtype=torch.float32) -> DataLoader:
        """
        获取回归任务的深度学习数据集。
        :param batch_size: 用于训练的 DataLoader 批次大小。
        :param eval_batch_size: 用于评估的 DataLoader 批次大小，默认为 batch_size。该参数仅在交叉验证中生效。
        :param start: 起始索引例（浮点数）。
        :param end: 结束索引比例（浮点数）。
        :param sample_gap: 采样间隔，默认为 1，表示连续采样；如果设置为大于 1 的值，表示每隔 sample_gap 个样本采样一个。
        :param shuffle: 是否打乱数据集，默认为 False。
        :param is_norm: 返回的结果是否进行归一化处理。
        :param fold: 交叉验证的折数，默认为 1 表示不使用交叉验证，此时参数 k 不起作用。
        :param k: 交叉验证的第 k 折，值应当大于等于 1 小于等于 fold，默认为 1。
        :param dtype: 转换后的数据类型，默认为 torch.float32。
        :return: 深度学习数据集 DataLoader，由 feature 和 target 两部分构成。
        Note:
            1. 如果使用交叉验证，那么 regdl_dataloader 为一个长度为 3 的元组，第一个值为用于训练的训练集，第二个值为用于评估的训练集，第三个值为验证集。
            2. 如果不使用交叉验证，那么只返回介于 start 和 end 之间的数据。
            3. 如果使用交叉验证，那么 shuffle 只会影响用于训练的训练集，不会影响用于验证的数据集。
        """
        if eval_batch_size is None:
            eval_batch_size = batch_size
        split_feature, split_target = self.get_regml(start=start, end=end, sample_gap=sample_gap, is_norm=is_norm,
                                                     to_numpy=True, fold=fold, k=k, dtype=np.float32)
        if isinstance(split_feature, tuple):  # 交叉验证。split_feature, split_target 数据类型相同。
            split_feature = tuple([torch.tensor(feature, dtype=dtype) for feature in split_feature])
            split_target = tuple([torch.tensor(target, dtype=dtype) for target in split_target])
            regdl_dataset = [TensorDataset(split_feature[0], split_target[0]),
                             TensorDataset(split_feature[1], split_target[1]),
                             TensorDataset(split_feature[2], split_target[2])]
            regdl_dataloader = tuple(
                [DataLoader(dataset=regdl_dataset[0], batch_size=batch_size, shuffle=shuffle, drop_last=False),
                 DataLoader(dataset=regdl_dataset[1], batch_size=eval_batch_size, shuffle=False, drop_last=False),
                 DataLoader(dataset=regdl_dataset[2], batch_size=eval_batch_size, shuffle=False, drop_last=False)]
            )
        else:
            split_feature, split_target = torch.tensor(split_feature, dtype=dtype), torch.tensor(split_target, dtype=dtype)
            # 创建 TensorDataset 和 DataLoader
            regdl_dataset = TensorDataset(split_feature, split_target)
            regdl_dataloader = DataLoader(
                dataset=regdl_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
            )
        return regdl_dataloader

    def get_seqdl(self, batch_size: int, eval_batch_size: int=None, start: float = 0.0, end: float = 1.0,
                  sample_gap: int=1, shuffle: bool=False, is_norm: bool=True, fold=1, k=1, dtype=torch.float32) -> DataLoader:
        """
        获取时序任务的深度学习数据集。
        :param batch_size: 用于训练的 DataLoader 批次大小。
        :param eval_batch_size: 用于评估的 DataLoader 批次大小，默认为 batch_size。该参数仅在交叉验证中生效。
        :param start: 起始索引比例（浮点数）。
        :param end: 结束索引比例（浮点数）。
        :param sample_gap: 采样间隔，默认为 1，表示连续采样；如果设置为大于 1 的值，表示每隔 sample_gap 个样本采样一个。
        :param shuffle: 是否打乱数据集，默认为 False。
        :param is_norm: 返回的结果是否进行归一化处理。
        :param fold: 交叉验证的折数，默认为 1 表示不使用交叉验证，此时参数 k 不起作用。
        :param k: 交叉验证的第 k 折，值应当大于等于 1 小于等于 fold，默认为 1。
        :param dtype: 转换后的数据类型，默认为 torch.float32。
        :return: 深度学习数据集 DataLoader，由 feature 和 target 两部分构成。
        Note:
            1. 输入特征数 = 时变已知特征数 + 时变未知特征数；样本数 = (数据长度 - 目标个数 - 时间步 + 1) // 采样间隔。
            2. 特征的尺寸为 [样本数，时间步，(输入特征数+目标个数)]。
            3. 目标的尺寸为 [样本数，目标个数]。
            4. 如果使用交叉验证，那么 seqdl_dataloader 为一个长度为 3 的元组，第一个值为用于训练的训练集，第二个值为用于评估的训练集，第三个值为验证集。
            5. 如果不使用交叉验证，那么只返回介于 start 和 end 之间的数据。
            6. 如果使用交叉验证，那么 shuffle 只会影响训练数据，不会影响验证数据。
        """
        if eval_batch_size is None:
            eval_batch_size = batch_size
        # 处理 k 折交叉验证
        k = 1 if fold == 1 else k
        fold_ratio = 1 / fold * (end - start)
        eval_start, eval_end = (k - 1) * fold_ratio + start, k * fold_ratio + start
        # 处理非标准化数据和标准化数据
        if is_norm and self.normalization:
            feature_seq, target_seq = self.feature_seq_norm, self.target_seq_norm
        elif not is_norm:
            feature_seq, target_seq = self.feature_seq, self.target_seq
        else:
            raise ValueError("未执行归一化操作，但是 is_norm=True")
        # 划分数据集
        train_feature_1, train_target_1 = split_sample(
            feature=feature_seq, target=target_seq, start_position=start, end_position=eval_start
        )
        train_feature_2, train_target_2 = split_sample(
            feature=feature_seq, target=target_seq, start_position=eval_end, end_position=end
        )
        eval_feature, eval_target = split_sample(
            feature=feature_seq, target=target_seq, start_position=eval_start, end_position=eval_end
        )
        train_feature = torch.concat([train_feature_1, train_feature_2], dim=0)
        train_target = torch.concat([train_target_1, train_target_2], dim=0)
        if train_feature.numel() == 0 and train_target.numel() == 0:  # 未使用交叉验证（训练特征和目标没有元素）
            # 下采样
            feature_sample = eval_feature[::sample_gap]
            target_sample = eval_target[::sample_gap]
            # 转换数据类型
            feature_sample, target_sample = feature_sample.type(dtype), target_sample.type(dtype)
            # 创建 TensorDataset 和 DataLoader
            seqdl_dataset = TensorDataset(feature_sample, target_sample)
            seqdl_dataloader = DataLoader(
                dataset=seqdl_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False
            )
        else:  # 使用交叉验证（训练特征和目标有元素）
            # 下采样
            train_feature_sample = train_feature[::sample_gap]
            train_target_sample = train_target[::sample_gap]
            # 转换数据类型
            train_feature_sample, train_target_sample = train_feature_sample.type(dtype), train_target_sample.type(dtype)
            train_feature, train_target = train_feature.type(dtype), train_target.type(dtype)
            eval_feature, eval_target = eval_feature.type(dtype), eval_target.type(dtype)
            # 创建 TensorDataset 和 DataLoader
            seqdl_dataset = [TensorDataset(train_feature_sample, train_target_sample),
                             TensorDataset(train_feature, train_target),
                             TensorDataset(eval_feature, eval_target)]
            seqdl_dataloader = tuple(
                [DataLoader(dataset=seqdl_dataset[0], batch_size=batch_size, shuffle=shuffle, drop_last=False),
                 DataLoader(dataset=seqdl_dataset[1], batch_size=eval_batch_size, shuffle=False, drop_last=False),
                 DataLoader(dataset=seqdl_dataset[2], batch_size=eval_batch_size, shuffle=False, drop_last=False)]
            )
        return seqdl_dataloader

    def __setitem__(self, key, value):
        """
        设置属性值。
        :param key: 属性名。
        :param value: 属性值。
        :return: None
        """
        setattr(self, key, value)


class ReadResult:
    def __init__(self, folder: str, time_step, output_size, writer):
        """
        读取基学习器的预测结果和特征值，并处理成指定的机器学习或深度学习数据格式。
        :param folder: 基学习器结果所在的目录（需要有 .identify 识别文件）
        :param time_step: 时间步长。
        :param output_size: 预测时间步长。
        :param writer: 实例化后的 Writer 对象。
        """
        self.result_dir, self.writer = folder, writer
        self.time_step, self.output_size = time_step, output_size
        self.selector = None  # 模型选择器
        self.time_dataset_path = None  # 处理后的时序数据集保存路径
        # 检查 .identify 文件
        check_result_identify(folder)
        # 读取 package.pkl 文件
        package_path = os.path.join(self.result_dir, 'data', 'package.pkl')
        if os.path.isfile(package_path):
            self.package = read_pickle(package_path)
        else:
            raise FileNotFoundError(f"未找到数据集封装文件：{package_path}，请检查文件路径以及基学习器保存目录是否正确！")
        # 提取 package.pkl 文件中的参数
        self.time_known_variables = self.package.dataset.time_known_variables  # 时变已知变量，数据类型均为 tuple。
        self.time_unknown_variables = self.package.dataset.time_unknown_variables  # 时变未知变量，数据类型均为 tuple。
        self.feature_variables = self.package.dataset.selected_feature_variables  # 全部特征变量，数据类型均为 tuple。
        self.target_variables = self.package.dataset.target_variables  # 目标变量，数据类型均为 tuple。
        self.time_target = self.package.dataset.time_target  # 时序目标，数据类型为 pd.DataFrame。用于验证预测结果的长度是否匹配。
        self.time_data = None  # 处理后的时间序列表

        # 计算训练集、验证集和测试集占比，以及样本数
        self.train_sample, self.valid_sample, self.test_sample, self.sample_number = None, None, None, None
        self.train_start, self.valid_start, self.test_start = None, None, None
        self.train_end, self.valid_end, self.test_end = None, None, None

        # 读取 fold metrics.xlsx 文件，获取基学习器的模型名称
        fold_metrics_path = os.path.join(self.result_dir, 'results', 'fold metrics.xlsx')
        if os.path.isfile(fold_metrics_path):
            fold_metrics = pd.read_excel(fold_metrics_path, index_col=0)
        else:
            raise FileNotFoundError(f"未找到基学习器交叉验证的评估指标文件：{fold_metrics_path}，请检查文件路径以及基学习器保存目录是否正确！")
        self.model_variables = tuple(fold_metrics.index.drop('Persistence'))  # 初始化选择后的模型变量，数据类型为 tuple。
        # 预测结果
        self.fold_predict, self.train_predict, self.valid_predict, self.test_predict = None, None, None, None  # Dict
        self.fold_predict_df, self.valid_predict_df, self.test_predict_df, self.all_predict_df = None, None, None, None  # pd.DataFrame
        self.fold_target_df, self.valid_target_df, self.test_target_df, self.all_target_df = None, None, None, None  # pd.DataFrame

        # 读取预测结果（字典类型，key 是目标名，value 是该目标名的预测结果（包含是预测时间步））
        self.fold_predict, self.train_predict, self.valid_predict, self.test_predict = dict(), dict(), dict(), dict()
        persistence_columns = [f'Persistence_{i}' for i in range(1, self.output_size + 1)]
        for target_name in self.target_variables:
            fold_predict = pd.read_excel(
                os.path.join(self.result_dir, 'results', f'fold predict ({target_name}).xlsx'), index_col=0
            ).reset_index(drop=True)
            train_predict = pd.read_excel(
                os.path.join(self.result_dir, 'results', f'train predict ({target_name}).xlsx'), index_col=0
            ).reset_index(drop=True)
            valid_predict = pd.read_excel(
                os.path.join(self.result_dir, 'results', f'valid predict ({target_name}).xlsx'), index_col=0
            ).reset_index(drop=True)
            test_predict = pd.read_excel(
                os.path.join(self.result_dir, 'results', f'test predict ({target_name}).xlsx'), index_col=0
            ).reset_index(drop=True)
            self.fold_predict[target_name] = fold_predict.drop(columns=persistence_columns, errors='ignore')
            self.train_predict[target_name] = train_predict.drop(columns=persistence_columns, errors='ignore')
            self.valid_predict[target_name] = valid_predict.drop(columns=persistence_columns, errors='ignore')
            self.test_predict[target_name] = test_predict.drop(columns=persistence_columns, errors='ignore')

    def set_feature(self, feature: Union[List[str], Tuple[str]]=None):
        """
        获取特征值，从基学习器封装的特征中选择参与集成的特征，要求所选择的特征必须在基学习器封装的数据中存在。
        :param feature: 特征名的列表或元组。默认为 None，表示使用所有的特征。
        :return: None
        """
        if feature is not None:
            assert isinstance(feature, (list, tuple)), "feature 必须是一个列表或元组！"
            assert all(isinstance(col, str) for col in feature), "feature 中的元素必须是字符串！"
            assert all(col in self.feature_variables for col in feature), "feature 中的元素必须在特征变量中存在！"
            feature_variables = []
            for col in feature:
                if col not in self.feature_variables:
                    warnings.warn(f"特征 '{col}' 不在数据集中，已忽略！")
                else:
                    feature_variables.append(col)
            self.feature_variables = tuple(feature_variables)  # 将变量名列表转换为元组
            self.time_known_variables = tuple([var for var in self.feature_variables if var in self.time_known_variables])
            self.time_unknown_variables = tuple([var for var in self.feature_variables if var in self.time_unknown_variables])

    def set_target(self, target: Union[List[str], Tuple[str]]=None):
        """
        获取目标值，选择需要预测的目标变量，要求所选择的变量必须在基学习器封装的数据和预测结果中存在。
        :param target: 目标名的特征或元组。默认为 None，表示使用所有的目标。
        :return: None
        """
        if target is not None:
            assert isinstance(target, (list, tuple)), "target 必须是一个列表或元组！"
            assert all(isinstance(col, str) for col in target), "target 中的元素必须是字符串！"
            assert all(col in self.target_variables for col in target), "target 中的元素必须在目标变量中存在！"
            target_variables = []
            for col in target:
                if col not in self.target_variables:
                    warnings.warn(f"目标 '{col}' 不在数据集中，已忽略！")
                else:
                    target_variables.append(col)
            self.target_variables = tuple(target_variables)  # 将变量名列表转换为元组

    def set_model(self, model: Union[List[str], Tuple[str]]=None):
        """
        获取预测值，从基学习器预测结果中选择参与集成的模型，要求所选择的预测模型必须在基学习器的预测结果中存在。
        :param model: 模型名的列表或元组。默认为 None，表示使用所有的预测结果。
        :return: None
        """
        if model is not None:
            assert isinstance(model, (list, tuple)), "model 必须是一个列表或元组！"
            assert all(isinstance(col, str) for col in model), "model 中的元素必须是字符串！"
            assert all(col in self.model_variables for col in model), "model 中的元素必须在模型变量中存在！"
            model_variables = []
            for col in model:
                if col not in self.model_variables:
                    warnings.warn(f"模型 '{col}' 不在数据集中，已忽略！")
                else:
                    model_variables.append(col)
            self.model_variables = tuple(model_variables)  # 将变量名列表转换为元组

    def select_model(self, method, number, how: Literal['inner', 'outer'] = 'outer'):
        """
        执行特征选择，并对 self.selected_feature 进行赋值（如果需要自定义可以手动赋值）。
        :param method: 特征选择方法。可选值为 None、'互信息'、'F检验'、'卡方检验'、'相关系数'（统计学筛选）；
                        以及 'sMAPE'、'MAPE'、'MSE'、'RMSE'、'MAE'、'R2'（指标筛选）。
        :param number: 每个目标列特征选择后特征的数量。由于不同目标列选择出来的结果可能不同，所以最终筛选出来的特征数量可能会不等于 number。
        :param how: 特征选择的合并方式。可选值为 'inner' 和 'outer'，分别表示交集和并集。
        :return: None
        Note:
            1. 特征选择会将训练集交叉验证预测结果、验证集预测结果和测试集预测结果拼接起来后再进行选择。
            2. 特征选择会将所有时间步长的预测结果拼接起来后再进行选择。
        """
        print(f"正在使用 {method} 方法选择 {number} 个模型，并且使用 {how} 方式进行合并...")
        self.writer.add_text(f"使用 {method} 方法选择 {number} 个模型，并且使用 {how} 方式进行合并。",
                             filename="Logs", folder="documents", suffix="log", save_mode='a+')
        self.selector = Selector(method=method, number=number)
        # 拼接数据集预测结果
        predict_result = dict()
        for target_name in self.target_variables:
            fold_predict = self.fold_predict[target_name]
            valid_predict = self.valid_predict[target_name]
            test_predict = self.test_predict[target_name]
            predict_concat = pd.concat([fold_predict, valid_predict, test_predict], axis=0).reset_index(drop=True)
            predict_reshape = pd.DataFrame()  # 重置预测结果
            for model_name in [target_name, ] + list(self.model_variables):
                predict_temp = pd.DataFrame(
                    predict_concat.filter(regex=f"^{model_name}_-").to_numpy().transpose(1, 0).reshape(-1,), columns=[model_name]
                )
                predict_reshape = pd.concat([predict_reshape, predict_temp], axis=1)
            predict_result[target_name] = predict_reshape
        # 模型选择
        model_selected_variables = None
        for target_name, value in predict_result.items():
            model_selected = self.selector.fit_transform(
                feature=value.drop(columns=[target_name]), target=value[target_name]
            )  # 选择模型
            model_selected_columns = model_selected.columns.tolist()
            if not model_selected_variables:
                model_selected_variables = model_selected_columns
            else:
                if how == 'inner':
                    model_selected_variables = list(set(model_selected_variables) & set(model_selected_columns))
                elif how == 'outer':
                    model_selected_variables = list(set(model_selected_variables) | set(model_selected_columns))
                else:
                    raise ValueError("how 参数只能为 'inner' 或 'outer'！")
        self.model_variables = tuple(model_selected_variables)  # 更新 self.model_variables 为选择后的模型变量

    def convert(self):
        # 1. 处理预测结果（根据选择后的目标和模型，提取选择后的预测结果并转为 DataFrame）
        self.fold_target_df, self.valid_target_df, self.test_target_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.fold_predict_df, self.valid_predict_df, self.test_predict_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for target_name in self.target_variables:
            for model_name in self.model_variables:
                fold_predict = self.fold_predict[target_name].filter(regex=f"^{model_name}_-")
                fold_predict = fold_predict.rename(columns=lambda x: f"{target_name}_{x}")
                valid_predict = self.valid_predict[target_name].filter(regex=f"^{model_name}_-")
                valid_predict = valid_predict.rename(columns=lambda x: f"{target_name}_{x}")
                test_predict = self.test_predict[target_name].filter(regex=f"^{model_name}_-")
                test_predict = test_predict.rename(columns=lambda x: f"{target_name}_{x}")
                self.fold_predict_df = pd.concat([self.fold_predict_df, fold_predict], axis=1)
                self.valid_predict_df = pd.concat([self.valid_predict_df, valid_predict], axis=1)
                self.test_predict_df = pd.concat([self.test_predict_df, test_predict], axis=1)
            fold_target = self.fold_predict[target_name].filter(regex=f"^{target_name}_-")
            valid_target = self.valid_predict[target_name].filter(regex=f"^{target_name}_-")
            test_target = self.test_predict[target_name].filter(regex=f"^{target_name}_-")
            self.fold_target_df = pd.concat([self.fold_target_df, fold_target], axis=1)
            self.valid_target_df = pd.concat([self.valid_target_df, valid_target], axis=1)
            self.test_target_df = pd.concat([self.test_target_df, test_target], axis=1)
        self.all_target_df = pd.concat([self.fold_target_df, self.valid_target_df, self.test_target_df], axis=0).reset_index(drop=True)
        self.all_predict_df = pd.concat([self.fold_predict_df, self.valid_predict_df, self.test_predict_df], axis=0).reset_index(drop=True)
        assert self.time_target.shape[0] == self.all_target_df.shape[0] + self.time_step + self.output_size - 1, \
            "读取的时间序列长度与预测长度不匹配"
        # 2. 获取新的 data 表（self.time_data），相较于原表，在前面少了 time_step + output_size - 1 行，在后面少了 output_size - 1 行。
        # 处理原始特征：仅取原 data 的第 time_step + output_size - 1 到 len(data) - output_size + 1（不包含结束位置）的索引。
        package_data = self.package.dataset.data.iloc[self.time_step + self.output_size - 1: len(self.package.dataset.data) - self.output_size + 1, :].reset_index(drop=True)
        # 同步处理目标（用于验证是否特征对齐）和预测（作为新特征添加到新表），偏移当前预测未来步-1个单位。
        target_data, target_max_shift = pd.DataFrame(), 0
        for col in self.all_target_df.columns:
            shift_step = int(col.rsplit('_-', 1)[1]) - 1
            target_max_shift = shift_step if shift_step > target_max_shift else target_max_shift
            if shift_step == 0:
                target_data = pd.concat(
                    [target_data, pd.DataFrame(self.all_target_df[[col]].values, columns=[col.rsplit('_-', 1)[0]])], axis=1
                )
        predict_data, predict_max_shift = pd.DataFrame(), 0
        for col in self.all_predict_df.columns:
            shift_step = int(col.rsplit('_-', 1)[1]) - 1
            predict_max_shift = shift_step if shift_step > predict_max_shift else predict_max_shift
            predict_data = pd.concat([predict_data, self.all_predict_df[[col]].shift(shift_step)], axis=1)
        target_data = target_data.iloc[target_max_shift:, :].reset_index(drop=True)  # 剪裁对齐预测结果
        predict_data = predict_data.iloc[predict_max_shift:, :].reset_index(drop=True)  # 剪裁不完整的预测数据
        predict_data.rename(columns=lambda x: x.split('_')[0] + '(' + x.split('_')[1] + ')' + '&'.join(x.split('_')[2:]), inplace=True)
        # 验证特征是否对齐，并将预测结果拼接到原特征中
        assert target_max_shift == predict_max_shift, "目标的偏移量和预测结果的偏移量不一致！"
        assert np.allclose(target_data.values, package_data[target_data.columns].values), "目标、预测结果和特征对齐失败！"
        self.time_data = pd.concat([package_data, predict_data], axis=1)
        # 3. 更新特征。
        self.time_known_variables = self.time_known_variables + tuple(predict_data.columns)
        # 4. 保存新的 data 表，即 self.time_data。
        self.time_dataset_path = os.path.join(self.writer.save_dir, 'data', 'time dataset.csv')  # 保存路径
        self.writer.add_df(self.time_data, filename='time dataset', folder='data', suffix='csv', save_mode='w+')
        # 5. 计算训练集、验证集和测试集占比，以及样本数。
        self.sample_number = self.time_data.shape[0] - self.time_step - self.output_size + 1
        self.train_sample = self.fold_predict_df.shape[0] - target_max_shift - self.time_step  # 保证与预测结果裁剪后的长度保持一致。
        self.valid_sample = self.valid_predict_df.shape[0]
        self.test_sample = self.test_predict_df.shape[0] - self.output_size + 1
        assert self.sample_number == self.train_sample + self.valid_sample + self.test_sample, "切分数据集的样本数不一致！"
        # 计算训练集、验证集和测试集的起始和结束位置
        self.train_start, self.train_end = 0.0, self.train_sample/self.sample_number
        self.valid_start, self.valid_end = self.train_end, (self.train_sample + self.valid_sample)/self.sample_number
        self.test_start, self.test_end = self.valid_end, 1.0
        del self.package  # 删除 package.pkl 文件中的数据，释放内存

    def __setitem__(self, key, value):
        """
        设置属性值。
        :param key: 属性名。
        :param value: 属性值。
        :return: None
        """
        setattr(self, key, value)


class PackageDataset:
    def __init__(self, dataset, train_start, train_end, valid_start, valid_end, test_start, test_end, train_batch_size,
                 eval_batch_size, sample_gap, k_fold, shuffle, time_step, output_size):
        """
        封装数据集。
        :param dataset: 实例后的 ReadDataset。
        :param train_start: 训练集起始索引比例（浮点数）。
        :param train_end: 训练集结束索引比例（浮点数）。
        :param valid_start: 验证集起始索引比例（浮点数）。
        :param valid_end: 验证集结束索引比例（浮点数）。
        :param test_start: 测试集起始索引比例（浮点数）。
        :param test_end: 测试集结束索引比例（浮点数）。
        :param train_batch_size: 训练过程批次大小。
        :param eval_batch_size: 验证过程批次大小。
        :param sample_gap: 采样间隔，默认为 1，表示连续采样；如果设置为大于 1 的值，表示每隔 sample_gap 个样本采样一个。
        :param k_fold: 交叉验证的折数，默认为 1 表示不使用交叉验证。
        :param shuffle: 是否打乱数据集，默认为 False。
        :param time_step: 时间步长。
        :param output_size: 目标值的个数。
        """
        # 保存输入参数
        self.dataset = dataset
        self.train_start, self.train_end = train_start, train_end
        self.valid_start, self.valid_end = valid_start, valid_end
        self.test_start, self.test_end = test_start, test_end
        self.train_batch_size, self.eval_batch_size = train_batch_size, eval_batch_size
        self.sample_gap, self.k_fold, self.shuffle = sample_gap, k_fold, shuffle
        self.time_step, self.output_size = time_step, output_size

        # 获取参数
        self.normalization = dataset.normalization
        # 获取数据集的特征变量和目标变量
        self.target_variables = list(dataset.target_variables)  # 长度等于目标数
        self.target_columns = list(dataset.target_columns)  # 长度等于预测步数乘以预测目标数，等于预测模型输出个数
        self.feature_seq_columns = list(dataset.feature_seq_columns)  # 时序数据集的列名。
        self.feature_reg_columns = list(dataset.feature_reg_columns)  # 回归数据集的列名。
        self.input_size_reg, self.input_size_seq = dataset.input_size_reg, dataset.input_size_seq

        # 机器学习评估器
        self.train_feature, self.valid_feature, self.test_feature = None, None, None  # 未标准化的特征（pd.DataFrame）
        self.train_target, self.valid_target, self.test_target = None, None, None  # 未标准化的目标（pd.DataFrame）
        if self.normalization:
            self.normalization_feature, self.normalization_target = dataset.normalization_feature, dataset.normalization_target
            self.train_feature_norm, self.valid_feature_norm, self.test_feature_norm = None, None, None  # 标准化的特征（pd.DataFrame）
            self.train_target_norm, self.valid_target_norm, self.test_target_norm = None, None, None  # 标准化的目标（pd.DataFrame）

        # 深度学习评估器
        self.train_evaler_reg, self.valid_evaler_reg, self.test_evaler_reg = None, None, None
        self.train_evaler_seq, self.valid_evaler_seq, self.test_evaler_seq = None, None, None
        if self.normalization:
            self.train_evaler_norm_reg, self.valid_evaler_norm_reg, self.test_evaler_norm_reg = None, None, None
            self.train_evaler_norm_seq, self.valid_evaler_norm_seq, self.test_evaler_norm_seq = None, None, None

        # 交叉验证训练器（Tuple[Tuple[Union[DataLoader, DataFrame]]]，第一个维度为折数+1，第二个维度分别为训练数据和验证数据）。
        # 如果 k_fold>1，那么为一个长度为 k_fold 的元组，每个元素为 k 折数据构成的训练集。
        # 如果 k_fold=1，那么为一个 tuple，里面只有一个 DataLoader 或 DataFrame。
        self.train_trainer_feature_fold, self.train_trainer_target_fold = None, None  # 回归任务机器学习训练器（未标准化）
        self.train_trainer_reg_fold, self.train_trainer_seq_fold = None, None  # 回归任务深度学习训练器和序列任务深度学习训练器（未标准化）
        if self.normalization:
            self.train_trainer_norm_feature_fold, self.train_trainer_norm_target_fold = None, None
            self.train_trainer_norm_reg_fold, self.train_trainer_norm_seq_fold = None, None

        # 总数据训练器，为 DataFrame 或 DataLoader 数据类型。
        self.train_trainer_feature_all, self.train_trainer_target_all = None, None  # 回归任务机器学习训练器（未标准化）
        self.train_trainer_reg_all, self.train_trainer_seq_all = None, None  # 回归任务深度学习训练器和序列任务深度学习训练器（未标准化）
        if self.normalization:
            self.train_trainer_norm_feature_all, self.train_trainer_norm_target_all = None, None
            self.train_trainer_norm_reg_all, self.train_trainer_norm_seq_all = None, None

        self.package_dataset()  # 封装数据集
        self.check_dataset()  # 检查数据集

    def package_dataset(self):
        # 数据集封装
        # 非标准化、回归任务、机器学习 数据集封装
        self.train_trainer_feature_fold, self.train_trainer_target_fold = [], []
        for k in range(1, self.k_fold + 1):
            feature, target = self.dataset.get_regml(
                start=self.train_start, end=self.train_end, sample_gap=self.sample_gap, is_norm=False, fold=self.k_fold, k=k
            )
            self.train_trainer_feature_fold.append(feature)
            self.train_trainer_target_fold.append(target)
        self.train_trainer_feature_fold = tuple(self.train_trainer_feature_fold)
        self.train_trainer_target_fold = tuple(self.train_trainer_target_fold)  # 构建交叉验证训练器
        self.train_trainer_feature_all, self.train_trainer_target_all = \
            self.dataset.get_regml(start=self.train_start, end=self.train_end, sample_gap=self.sample_gap, is_norm=False)  # 构建总数据训练器
        self.train_feature, self.train_target = self.dataset.get_regml(start=self.train_start, end=self.train_end, is_norm=False)
        self.valid_feature, self.valid_target = self.dataset.get_regml(start=self.valid_start, end=self.valid_end, is_norm=False)
        self.test_feature, self.test_target = self.dataset.get_regml(start=self.test_start, end=self.test_end, is_norm=False)  # 构建评估器
        # 非标准化、回归任务、深度学习 数据集封装
        self.train_trainer_reg_fold = tuple([self.dataset.get_regdl(
            batch_size=self.train_batch_size, eval_batch_size=self.eval_batch_size, start=self.train_start,
            end=self.train_end, sample_gap=self.sample_gap, shuffle=self.shuffle, is_norm=False, fold=self.k_fold, k=k
        ) for k in range(1, self.k_fold + 1)])  # 构建交叉验证训练器
        self.train_trainer_reg_all = self.dataset.get_regdl(
            batch_size=self.train_batch_size, start=self.train_start, end=self.train_end, sample_gap=self.sample_gap,
            shuffle=self.shuffle, is_norm=False
        )  # 构建总数据训练器
        self.train_evaler_reg = self.dataset.get_regdl(
            batch_size=self.eval_batch_size, start=self.train_start, end=self.train_end, sample_gap=1,
            shuffle=False, is_norm=False
        )
        self.valid_evaler_reg = self.dataset.get_regdl(
            batch_size=self.eval_batch_size, start=self.valid_start, end=self.valid_end, sample_gap=1,
            shuffle=False, is_norm=False
        )
        self.test_evaler_reg = self.dataset.get_regdl(
            batch_size=self.eval_batch_size, start=self.test_start, end=self.test_end, sample_gap=1,
            shuffle=False, is_norm=False
        )
        # 非标准化、序列任务、深度学习 数据集封装
        self.train_trainer_seq_fold = tuple([self.dataset.get_seqdl(
            batch_size=self.train_batch_size, eval_batch_size=self.eval_batch_size, start=self.train_start,
            end=self.train_end, sample_gap=self.sample_gap, shuffle=self.shuffle, is_norm=False, fold=self.k_fold, k=k
        ) for k in range(1, self.k_fold + 1)])  # 构建交叉验证训练器
        self.train_trainer_seq_all = self.dataset.get_seqdl(
            batch_size=self.train_batch_size, start=self.train_start, end=self.train_end, sample_gap=self.sample_gap,
            shuffle=self.shuffle, is_norm=False
        )  # 构建总数据训练器
        self.train_evaler_seq = self.dataset.get_seqdl(
            batch_size=self.eval_batch_size, start=self.train_start, end=self.train_end, sample_gap=1,
            shuffle=False, is_norm=False
        )
        self.valid_evaler_seq = self.dataset.get_seqdl(
            batch_size=self.eval_batch_size, start=self.valid_start, end=self.valid_end, sample_gap=1,
            shuffle=False, is_norm=False
        )
        self.test_evaler_seq = self.dataset.get_seqdl(
            batch_size=self.eval_batch_size, start=self.test_start, end=self.test_end, sample_gap=1,
            shuffle=False, is_norm=False
        )
        if self.normalization:
            # 标准化、回归任务、机器学习 数据集封装
            self.train_trainer_norm_feature_fold, self.train_trainer_norm_target_fold = [], []
            for k in range(1, self.k_fold + 1):
                feature, target = self.dataset.get_regml(
                    start=self.train_start, end=self.train_end, sample_gap=self.sample_gap, is_norm=True, fold=self.k_fold, k=k
                )
                self.train_trainer_norm_feature_fold.append(feature)
                self.train_trainer_norm_target_fold.append(target)
            self.train_trainer_norm_feature_fold = tuple(self.train_trainer_norm_feature_fold)
            self.train_trainer_norm_target_fold = tuple(self.train_trainer_norm_target_fold)  # 构建交叉验证训练器
            self.train_trainer_norm_feature_all, self.train_trainer_norm_target_all = \
                self.dataset.get_regml(start=self.train_start, end=self.train_end, sample_gap=self.sample_gap, is_norm=True)  # 构建总数据训练器
            self.train_feature_norm, self.train_target_norm = self.dataset.get_regml(
                start=self.train_start, end=self.train_end, is_norm=True
            )
            self.valid_feature_norm, self.valid_target_norm = self.dataset.get_regml(
                start=self.valid_start, end=self.valid_end, is_norm=True
            )
            self.test_feature_norm, self.test_target_norm = self.dataset.get_regml(
                start=self.test_start, end=self.test_end, is_norm=True
            )
            # 标准化、回归任务、深度学习 数据集封装
            self.train_trainer_norm_reg_fold = tuple([self.dataset.get_regdl(
                batch_size=self.train_batch_size, eval_batch_size=self.eval_batch_size, start=self.train_start,
                end=self.train_end, sample_gap=self.sample_gap, shuffle=self.shuffle, is_norm=True, fold=self.k_fold, k=k
            ) for k in range(1, self.k_fold + 1)])  # 构建交叉验证训练器
            self.train_trainer_norm_reg_all = self.dataset.get_regdl(
                batch_size=self.train_batch_size, start=self.train_start, end=self.train_end, sample_gap=self.sample_gap,
                shuffle=self.shuffle, is_norm=True
            )  # 构建总数据训练器
            self.train_evaler_norm_reg = self.dataset.get_regdl(
                batch_size=self.eval_batch_size, start=self.train_start, end=self.train_end, sample_gap=1, shuffle=False, is_norm=True
            )
            self.valid_evaler_norm_reg = self.dataset.get_regdl(
                batch_size=self.eval_batch_size, start=self.valid_start, end=self.valid_end, sample_gap=1, shuffle=False, is_norm=True
            )
            self.test_evaler_norm_reg = self.dataset.get_regdl(
                batch_size=self.eval_batch_size, start=self.test_start, end=self.test_end, sample_gap=1, shuffle=False, is_norm=True
            )
            # 标准化、序列任务、深度学习 数据集封装
            self.train_trainer_norm_seq_fold = tuple([self.dataset.get_seqdl(
                batch_size=self.train_batch_size, eval_batch_size=self.eval_batch_size, start=self.train_start,
                end=self.train_end, sample_gap=self.sample_gap, shuffle=self.shuffle, is_norm=True, fold=self.k_fold, k=k
            ) for k in range(1, self.k_fold + 1)])  # 构建交叉验证训练器
            self.train_trainer_norm_seq_all = self.dataset.get_seqdl(
                batch_size=self.train_batch_size, start=self.train_start, end=self.train_end, sample_gap=self.sample_gap,
                shuffle=self.shuffle, is_norm=True
            )  # 构建总数据训练器
            self.train_evaler_norm_seq = self.dataset.get_seqdl(
                batch_size=self.eval_batch_size, start=self.train_start, end=self.train_end, sample_gap=1, shuffle=False, is_norm=True
            )
            self.valid_evaler_norm_seq = self.dataset.get_seqdl(
                batch_size=self.eval_batch_size, start=self.valid_start, end=self.valid_end, sample_gap=1, shuffle=False, is_norm=True
            )
            self.test_evaler_norm_seq = self.dataset.get_seqdl(
                batch_size=self.eval_batch_size, start=self.test_start, end=self.test_end, sample_gap=1, shuffle=False, is_norm=True
            )

    def check_dataset(self):
        # 回归任务数据集和时序任务数据集一致性检查
        sort_variables = []  # 对回归任务的特征进行排序，以保证和时序任务的特征位置保持一致
        tensor_variables = []  # 时序任务的列名
        tensor_position = []  # 选取时序任务的特征位置
        for i in range(self.time_step):  # 为时序数据集（tensor）赋予列名
            known_ind = self.time_step - self.output_size - i if self.time_step - self.output_size - i > 0 else self.time_step - self.output_size - i - 1
            tensor_variables.extend([f"{var}_{known_ind}" for var in self.dataset.time_known_variables])
            tensor_variables.extend([f"{var}_{self.time_step - i}" for var in self.dataset.time_unknown_variables + self.dataset.target_variables])
        if self.dataset.add_history_features:
            sort_variables = tensor_variables
            tensor_position = [True] * len(sort_variables)
        else:
            sort_variables_set = set(self.train_feature.columns)
            sort_variables = [var for var in tensor_variables if var in sort_variables_set]
            tensor_position = [var in sort_variables_set for var in tensor_variables]
        temp = self.train_evaler_seq.dataset.tensors[0].cpu().numpy()
        assert np.allclose(self.train_feature[sort_variables].values, temp.reshape(temp.shape[0], -1)[:, tensor_position]), \
            "回归任务训练集特征和时序任务训练集特征不一致！"
        temp = self.valid_evaler_seq.dataset.tensors[0].cpu().numpy()
        assert np.allclose(self.valid_feature[sort_variables].values, temp.reshape(temp.shape[0], -1)[:, tensor_position]), \
            "回归任务验证集特征和时序任务验证集特征不一致！"
        temp = self.test_evaler_seq.dataset.tensors[0].cpu().numpy()
        assert np.allclose(self.test_feature[sort_variables].values, temp.reshape(temp.shape[0], -1)[:, tensor_position]), \
            "回归任务测试集特征和时序任务测试集特征不一致！"
        del temp
        assert np.allclose(self.train_target.values, self.train_evaler_seq.dataset.tensors[1].cpu().numpy()), \
            "回归任务训练集目标和时序任务训练集目标不一致！"
        assert np.allclose(self.valid_target.values, self.valid_evaler_seq.dataset.tensors[1].cpu().numpy()), \
            "回归任务验证集目标和时序任务验证集目标不一致！"
        assert np.allclose(self.test_target.values, self.test_evaler_seq.dataset.tensors[1].cpu().numpy()), \
            "回归任务测试集目标和时序任务测试集目标不一致！"

    def write_data(self, writer):
        """
        将数据集写入文件（覆写）
        :param writer: 实例化后的 Writer 对象。
        :return: None
        """
        # 将真实值写入 results 文件（不同的预测目标分配不同的文件）
        for target_name in self.target_variables:
            writer.add_df(
                data_df=self.train_target.filter(regex=f"^{target_name}_-").set_index(pd.Index(range(1, len(self.train_target) + 1))),
                axis=1, filename=f"fold predict ({target_name})", folder="results", suffix='xlsx', save_mode='a+'
            )
            writer.add_df(
                data_df=self.train_target.filter(regex=f"^{target_name}_-").set_index(pd.Index(range(1, len(self.train_target) + 1))),
                axis=1, filename=f"train predict ({target_name})", folder="results", suffix='xlsx', save_mode='a+'
            )
            writer.add_df(
                data_df=self.valid_target.filter(regex=f"^{target_name}_-").set_index(pd.Index(range(1, len(self.valid_target) + 1))),
                axis=1, filename=f"valid predict ({target_name})", folder="results", suffix='xlsx', save_mode='a+'
            )
            writer.add_df(
                data_df=self.test_target.filter(regex=f"^{target_name}_-").set_index(pd.Index(range(1, len(self.test_target) + 1))),
                axis=1, filename=f"test predict ({target_name})", folder="results", suffix='xlsx', save_mode='a+'
            )
        # 将目标写入 data 目录下
        if self.normalization:
            columns = [col for col in self.train_target.columns] + [col + ' (standardized)' for col in self.train_target_norm.columns]
            train_target = pd.DataFrame(np.concatenate([self.train_target.values, self.train_target_norm.values], axis=1),
                                    columns=columns, index=range(1, len(self.train_target) + 1))
            valid_target = pd.DataFrame(np.concatenate([self.valid_target.values, self.valid_target_norm.values], axis=1),
                                    columns=columns, index=range(1, len(self.valid_target) + 1))
            test_target = pd.DataFrame(np.concatenate([self.test_target.values, self.test_target_norm.values], axis=1),
                                   columns=columns, index=range(1, len(self.test_target) + 1))
        else:
            columns = [col for col in self.train_target.columns]
            train_target = pd.DataFrame(self.train_target.values, columns=columns, index=range(1, len(self.train_target) + 1))
            valid_target = pd.DataFrame(self.valid_target.values, columns=columns, index=range(1, len(self.valid_target) + 1))
            test_target = pd.DataFrame(self.test_target.values, columns=columns, index=range(1, len(self.test_target) + 1))
        writer.add_df(data_df=train_target, axis=1, filename="train target", folder="data", save_mode='w+')
        writer.add_df(data_df=valid_target, axis=1, filename="valid target", folder="data", save_mode='w+')
        writer.add_df(data_df=test_target, axis=1, filename="test target", folder="data", save_mode='w+')
        # 将未标准化的特征写入 data 目录下
        writer.add_df(
            data_df=self.train_feature.set_index(pd.Index(range(1, len(self.train_feature) + 1))),
            axis=1, filename="train feature", folder="data", save_mode='w+'
        )
        writer.add_df(
            data_df=self.valid_feature.set_index(pd.Index(range(1, len(self.valid_feature) + 1))),
            axis=1, filename="valid feature", folder="data", save_mode='w+'
        )
        writer.add_df(
            data_df=self.test_feature.set_index(pd.Index(range(1, len(self.test_feature) + 1))),
            axis=1, filename="test feature", folder="data", save_mode='w+'
        )
        # 将标准化的特征写入 data 目录下
        if self.normalization:
            writer.add_df(
                data_df=self.train_feature_norm.set_index(pd.Index(range(1, len(self.train_feature_norm) + 1))),
                axis=1, filename="train feature (standardized)", folder="data", save_mode='w+'
            )
            writer.add_df(
                data_df=self.valid_feature_norm.set_index(pd.Index(range(1, len(self.valid_feature_norm) + 1))),
                axis=1, filename="valid feature (standardized)", folder="data", save_mode='w+'
            )
            writer.add_df(
                data_df=self.test_feature_norm.set_index(pd.Index(range(1, len(self.test_feature_norm) + 1))),
                axis=1, filename="test feature (standardized)", folder="data", save_mode='w+'
            )

    def write_predict(self, writer, value: pd.DataFrame, model: str, dataset: str):
        """
        将预测结果写入文件
        :param writer: 实例化后的 Writer 对象。
        :param value: 预测结果。
        :param model: 模型名称。
        :param dataset: 数据集名称：fold、train、valid、test。
        :return: None
        """
        for target_name in self.target_variables:
            data_df = value.filter(regex=f"^{target_name}_-").set_index(pd.Index(range(1, len(value) + 1)))
            data_df.columns = [col.replace(f"{target_name}_", f"{model}_") for col in data_df.columns]
            writer.add_df(data_df=data_df, axis=1, filename=f"{dataset} predict ({target_name})",
                          folder="results", suffix='xlsx', save_mode='a+')


class ReadParameter:
    def __init__(self):
        self.data, self.model = None, None  # 初始化数据参数和模型参数，数据类型均为字典。

    def read_data(self, path: str):
        self.data = read_yaml(path)
        return self.data

    def read_model(self, path: str):
        self.model = read_yaml(path)
        return self.model

    def write_parameter(self, writer):
        """
        将参数写入文件。
        :param writer: 实例化后的 Writer 对象。
        :return: None
        """
        writer.add_param(
            param_desc='[data parameters]',
            param_dict=self.data,
            filename='config parameters',
            folder='documents',
            save_mode='a+'
        )
        writer.add_param(
            param_desc='[model parameters]',
            param_dict=self.model,
            filename='config parameters',
            folder='documents',
            save_mode='a+'
        )

    def __getitem__(self, item):
        """
        获取属性值。
        :param item: 属性名。
        :return: 属性值。
        """
        return getattr(self, item)

    def __setitem__(self, key, value):
        """
        设置属性值。
        :param key: 属性名。
        :param value: 属性值。
        :return: None
        """
        setattr(self, key, value)

    def __iter__(self):
        return iter(self.__dict__.items())

    def items(self):
        return self.__dict__.items()



