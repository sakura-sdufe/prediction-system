# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/29 22:36
# @Author   : 张浩
# @FileName : scaler.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple


class Norm:
    """对数据进行使用 Norm 方法进行归一化和反归一化"""
    def __init__(self, data):
        """
        :param data: predict 可以是 2Darray, DataFrame, Series。对列进行归一化，即对 axis=0 方向进行归一化操作（有几列就执行几次归一化操作）。
            Note：2Darray 的数据类型应当为数值型，DataFrame 应当存在数值型的列。
            Note：标准化结果存放在 norm_result 属性中，可以通过 get_norm_result 方法获取。
        """
        self.data = data  # 保存输入数据
        self.mean = np.empty(0)  # 标准化均值
        self.std = np.empty(0)  # 标准化标准差
        self.norm_array = np.empty((0, 0))  # 标准化后 np.ndarray
        self.norm_result = None  # 标准化结果

        self.data_type = type(data)  # 记录输入类型
        self.type_ndim = data.ndim if isinstance(data, np.ndarray) else 0  # 记录输入维度
        self.array, self.index, self.names = _convert_2Dndarray(data)  # 转换为 2Darray

        self._norm_private()  # 对数据进行 Norm 操作
        self.norm_result = _convert_type(self.norm_array, typ=self.data_type, ndim=self.type_ndim,
                                         index=self.index, names=self.names)  # 转换为输出格式

    def _norm_private(self):
        """对数据进行归一化操作"""
        self.mean = np.mean(self.array, axis=0)
        self.std = np.std(self.array, axis=0)
        self.norm_array = (self.array - self.mean) / self.std

    def get_norm_result(self):
        """获取归一化后的数据"""
        return self.norm_result

    def norm(self, data, numbers:Union[List, Tuple, int]=None, columns:Union[List, Tuple, str]=None):
        """
        对给定的数据进行标准化操作。
        :param data: 你需要传入你想标准化的数据，支持 np.1Darray, np.2Darray, pd.Series, pd.DataFrame。
        :param numbers: 你可以传入整数，这个整数表示你所输入的数据使用第几列的参数进行标准化（默认为第 0 列）。
        :param columns: 你也可以输入一个列名（string），这样就可以使用列名进行标准化。
        :return: 返回标准化后的数据，数据类型与输入数据类型一致。
        Note:
            1. 如果不指定 number 和 columns 参数，将默认使用第 0 至 data.shape[1] 列进行标准化。
            2. 如果第二个参数不以键值对的形式传入，可以传入整数或者字符串，但是不能同时传入整数和字符串
                （也可以传入包含字符串或整数的列表或元组，但是深度最高为一维）。
            3. 参数 data 的列数应当和 numbers 或 columns 的长度相同，并且是有序的。
                如果 data 中所有列均采用相同的均值和标准差进行标准化，也可以只传入标准化参数的列数或列名。
        """
        # 参数检查
        assert not (numbers is not None and columns is not None), 'number 和 columns 参数只能传入一个！'
        if self.names.empty:  # 原始数据是 np.ndarray 类型
            info = '当原始数据是 np.ndarray 类型时，columns 参数无效。'
            assert (not isinstance(numbers, str)) and (not isinstance(columns, str)), info
            if isinstance(numbers, (List, Tuple)):
                assert (not isinstance(numbers[0], str)), info
            if isinstance(columns, (List, Tuple)):
                assert (not isinstance(columns[0], str)), info

        # 处理 data
        data_type = type(data)  # 记录输入类型
        type_ndim = data.ndim if isinstance(data, np.ndarray) else 0  # 记录输入维度
        data_array, data_index, data_names = _convert_2Dndarray(data)  # 处理输入数据
        need_cols = data_array.shape[1]  # 记录输入数据的列数
        # 解析参数
        if numbers is None and columns is None:  # 未指定列，默认使用第 0 至 data.shape[1]-1 列
            data_numbers = list(range(data_array.shape[1]))
        elif isinstance(columns, str):  # 以键值对的形式向 columns 传入参数，且传入的参数为字符串
            data_numbers = [self.names.get_loc(columns)] * need_cols
        elif isinstance(columns, (List, Tuple)):  # 以键值对的形式向 columns 传入参数，且传入的参数为列表或元组
            data_numbers = [self.names.get_loc(c) for c in columns]
        elif isinstance(numbers, int):  # 以键值对的形式向 numbers 传入参数，且传入的参数为整数
            data_numbers = [numbers] * need_cols
        elif isinstance(numbers, str):  # 以键值对的形式向 numbers 传入参数，且传入的参数为字符串
            data_numbers = [self.names.get_loc(numbers)] * need_cols
        elif isinstance(numbers, (List, Tuple)) and isinstance(numbers[0], int):  # 以键值对的形式向 numbers 传入参数，且传入的参数为列表或元组（每个元素都是整数）
            data_numbers = list(numbers)
        elif isinstance(numbers, (List, Tuple)) and isinstance(numbers[0], str):  # 以键值对的形式向 numbers 传入参数，且传入的参数为列表或元组（每个元素都是字符串）
            data_numbers = [self.names.get_loc(n) for n in numbers]
        else:
            raise ValueError('请输入合适的参数指定列。')

        # 对数据进行归一化操作
        need_mean = self.mean[data_numbers]  # 获取均值
        need_std = self.std[data_numbers]  # 获取标准差
        norm_array = (data_array - need_mean) / need_std  # 归一化
        # 转换为输出格式
        norm_result = _convert_type(norm_array, typ=data_type, ndim=type_ndim, index=data_index, names=data_names)
        return norm_result

    def denorm(self, data, numbers:Union[List, Tuple, int]=None, columns:Union[List, Tuple, str]=None):
        """
        对给定的数据进行反标准化操作。
        :param data: 你需要传入你想反标准化的数据，支持 np.1Darray, np.2Darray, pd.Series, pd.DataFrame。
        :param numbers: 你可以传入整数，这个整数表示你所输入的数据使用第几列的参数进行反标准化（默认为第 0 列）。
        :param columns: 你也可以输入一个列名（string），这样就可以使用列名进行反标准化。
        :return: 返回反标准化后的数据，数据类型与输入数据类型一致。
        Note:
            1. 如果不指定 number 和 columns 参数，将默认使用第 0 至 data.shape[1] 列进行标准化。
            2. 如果第二个参数不以键值对的形式传入，可以传入整数或者字符串，但是不能同时传入整数和字符串
                （也可以传入包含字符串或整数的列表或元组，但是深度最高为一维）。
            3. 参数 data 的列数应当和 numbers 或 columns 的长度相同，并且是有序的。
                如果 data 中所有列均采用相同的均值和标准差进行标准化，也可以只传入标准化参数的列数或列名。
        """
        # 参数检查
        assert not (numbers is not None and columns is not None), 'number 和 columns 参数只能传入一个！'
        if self.names.empty:  # 原始数据是 np.ndarray 类型
            info = '当原始数据是 np.ndarray 类型时，columns 参数无效。'
            assert (not isinstance(numbers, str)) and (not isinstance(columns, str)), info
            if isinstance(numbers, (List, Tuple)):
                assert (not isinstance(numbers[0], str)), info
            if isinstance(columns, (List, Tuple)):
                assert (not isinstance(columns[0], str)), info

        # 处理 data
        data_type = type(data)  # 记录输入类型
        type_ndim = data.ndim if isinstance(data, np.ndarray) else 0  # 记录输入维度
        data_array, data_index, data_names = _convert_2Dndarray(data)  # 处理输入数据
        # 解析参数
        if numbers is None and columns is None:  # 未指定列，默认使用第 0 至 data.shape[1]-1 列
            data_numbers = list(range(data_array.shape[1]))
        elif isinstance(columns, str):  # 以键值对的形式向 columns 传入参数，且传入的参数为字符串
            data_numbers = [self.names.get_loc(columns)]
        elif isinstance(columns, (List, Tuple)):  # 以键值对的形式向 columns 传入参数，且传入的参数为列表或元组
            data_numbers = [self.names.get_loc(c) for c in columns]
        elif isinstance(numbers, int):  # 以键值对的形式向 numbers 传入参数，且传入的参数为整数
            data_numbers = [numbers]
        elif isinstance(numbers, str):  # 以键值对的形式向 numbers 传入参数，且传入的参数为字符串
            data_numbers = [self.names.get_loc(numbers)]
        elif isinstance(numbers, (List, Tuple)) and isinstance(numbers[0], int):  # 以键值对的形式向 numbers 传入参数，且传入的参数为列表或元组（每个元素都是整数）
            data_numbers = list(numbers)
        elif isinstance(numbers, (List, Tuple)) and isinstance(numbers[0], str):  # 以键值对的形式向 numbers 传入参数，且传入的参数为列表或元组（每个元素都是字符串）
            data_numbers = [self.names.get_loc(n) for n in numbers]
        else:
            raise ValueError('请输入合适的参数指定列。')

        # 对数据进行反标准化操作
        need_mean = self.mean[data_numbers]  # 获取均值
        need_std = self.std[data_numbers]  # 获取标准差
        norm_array = data_array * need_std + need_mean  # 反标准化
        # 转换为输出格式
        norm_result = _convert_type(norm_array, typ=data_type, ndim=type_ndim, index=data_index, names=data_names)
        return norm_result


class MinMax:
    """对数据进行使用 MinMax 方法进行归一化和反归一化"""
    pass


def _convert_2Dndarray(data:Union[pd.Series, pd.DataFrame, np.ndarray]):
    if isinstance(data, np.ndarray) and np.ndim(data) == 1:
        data_array = np.expand_dims(data, axis=1)
        data_index, data_names = pd.Index([]), pd.Index([])
    elif isinstance(data, np.ndarray) and np.ndim(data) == 2:
        data_array = data
        data_index, data_names = pd.Index([]), pd.Index([])
    elif isinstance(data, pd.Series):
        data_array = data.to_frame().to_numpy()
        data_index, data_names = data.index, pd.Index([data.name])
    elif isinstance(data, pd.DataFrame):
        data_array = data.select_dtypes(include=[np.number]).to_numpy()
        data_index, data_names = data.index, data.select_dtypes(include=[np.number]).columns
    else:
        raise ValueError('predict 只能是 1Darray, 2Darray, DataFrame, Series 中的一种。')
    return data_array, data_index, data_names


def _convert_type(nd:np.ndarray, typ:type, ndim:int=0, index=None, names=None):
    """
    将 np.ndarray(2D) 转换为指定类型（pd.DataFrame、pd.Series、np.ndarray），如果转为 np.ndarray 需要指定维度。
    如果转为 np.ndarray，需要指定维度（ndim）；如果转为 pd.Series 或 pd.DataFrame，需要指定索引（index）和列名（names）。
    """
    assert isinstance(nd, np.ndarray) and (np.ndim(nd) == 2), 'nd 只能是 2Darray 类型。'
    if (typ is np.ndarray) and (ndim == 1):
        return nd.squeeze(axis=1)  # 转为 1Darray
    elif (typ is np.ndarray) and (ndim == 2):
        return nd  # 转为 2Darray
    elif typ is pd.Series:
        names = names if isinstance(names, str) else names[0]
        return pd.Series(nd.squeeze(), index=index, name=names)  # 转为 Series
    elif typ is pd.DataFrame:
        return pd.DataFrame(nd, index=index, columns=names)  # 转为 DataFrame
    else:
        raise ValueError('请输入正确的类型！typ 只能是 np.ndarray、pd.Series、pd.DataFrame 中的一种。')
