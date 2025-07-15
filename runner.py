# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/5/4 16:33
# @Author   : 张浩
# @FileName : runner.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

import re
import os
import time
import shutil
import traceback
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from sklearn import base
from copy import deepcopy
from typing import List, Dict, Tuple, Union

from base import BaseRegression, BaseSequence
from data import Norm, ReadDataset, ReadResult, ReadParameter, PackageDataset
from utils.metrics import calculate_metrics
from utils.cprint import cprint
from utils.writer import Writer

# 设置 DataFrame 显示选项
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)  # 设置显示宽度，避免换行
pd.set_option('display.colheader_justify', 'left')  # 列标题左对齐
# 恢复默认设置
# pd.reset_option('display.max_rows')
# pd.reset_option('display.max_columns')


class Runner:
    """运行器"""
    def __init__(self, data_parameter_path, model_parameter_path):
        # 初始化变量
        self.dataset = None  # ReadDataset 类
        self.parameter = ReadParameter()  # 初始化参数类
        self.package = None  # 数据封装类，实例化 PackageDataset 类
        self.error_number = 0  # 运行遇到的错误次数
        # 读取 data 和 model 参数
        self.parameter_data = self.parameter.read_data(data_parameter_path)  #  读取数据参数
        self.parameter_model = self.parameter.read_model(model_parameter_path)  # 读取模型参数
        # 提取参数
        self.save_mode = 'w+' if self.parameter_data['DeleteDir'] else 'a+'  # Writer 类写入本地模式。
        self.time_step, self.output_size = self.parameter_data['TimeStep'], self.parameter_data['OutputSize']
        self.sample_gap, self.shuffle = self.parameter_data['SampleGap'], self.parameter_data['Shuffle']
        self.train_batch_size, self.eval_batch_size = self.parameter_data['TrainBatchSize'], self.parameter_data['EvalBatchSize']
        self.k_fold = self.parameter_data['KFold']
        # 处理读取目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not self.parameter_data['EnsembleMethod']:  # 非集成过程，self.read_path 为数据集读取的路径
            self.read_path: str = os.path.join(current_dir, "datasets", self.parameter_data['FileName'])  # type: ignore
        else:  # 集成过程，self.read_path 为集成器读取的目录，即预测器保存结果的目录
            self.read_path: str = os.path.join(current_dir, self.parameter_data['ReadResultDir'].strip(r'./\\'))  # type: ignore
        # 处理保存目录 和 Writer 类
        if os.path.isabs(self.parameter_data['SaveResultDir']):
            self.save_dir = self.parameter_data['SaveResultDir']
        else:
            self.save_dir = os.path.join(current_dir, self.parameter_data['SaveResultDir'].strip(r'./\\'))
        self.writer = Writer(self.save_dir, is_delete=self.parameter_data['DeleteDir'])  # Writer 类，用于保存结果
        # 处理数据
        self.weights, self.weights_train = None, None  # 时间步权重 和 训练时间步权重
        self.train_target, self.valid_target, self.test_target = None, None, None
        self.train_persistence, self.valid_persistence, self.test_persistence = None, None, None  # Persistence 预测结果
        self.train_average, self.valid_average, self.test_average = None, None, None  # Average 集成预测结果
        self.average_columns = None  # Average 集成参与的列名
        self.process_data()

    def process_data(self):
        normalization = eval(self.parameter_data['ScaleMethod']) if self.parameter_data['ScaleMethod'] else None
        if not self.parameter_data['EnsembleMethod']:  # 非集成过程，self.read_path 为数据集读取的路径
            # 封装成 ReadDataset 类。
            self.dataset = ReadDataset(
                path=self.read_path, time_step=self.time_step, output_size=self.output_size, writer=self.writer,
                add_history_features=self.parameter_data['AddHistoryFeatures'], normalization=normalization
            )
            self.dataset.set_feature(
                time_known=self.parameter_data['TimeKnown'], time_unknown=self.parameter_data['TimeUnknown']
            )
            self.dataset.set_target(target=self.parameter_data['Target'])
            print("成功完成 特征和目标读取！")
            self.writer.add_text("成功完成 特征和目标读取！", filename="Logs", folder="documents", suffix="log")
            self.dataset.select_feature(
                method=self.parameter_data['FeatureSelectionMethod'],
                number=self.parameter_data['FeatureSelectionNumber'],
                how=self.parameter_data['FeatureSelectionHow']
            )
            print(f"选择特征：{self.dataset.selected_feature_variables}")
            self.writer.add_text(f"选择特征：{self.dataset.selected_feature_variables}",
                                 filename="Logs", folder="documents", suffix="log")
            self.dataset.convert()
            print("成功完成 特征转换！")
            self.writer.add_text("成功完成 特征转换！", filename="Logs", folder="documents", suffix="log")
            # 封装成 PackageDataset 类。
            train_start, train_end = self.parameter_data['TrainStartPosition'], self.parameter_data['TrainEndPosition']
            valid_start, valid_end = self.parameter_data['ValidStartPosition'], self.parameter_data['ValidEndPosition']
            test_start, test_end = self.parameter_data['TestStartPosition'], self.parameter_data['TestEndPosition']
            self.package = PackageDataset(
                dataset=self.dataset, train_start=train_start, train_end=train_end, valid_start=valid_start,
                valid_end=valid_end, test_start=test_start, test_end=test_end, train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size, sample_gap=self.sample_gap, k_fold=self.k_fold,
                shuffle=self.shuffle, time_step=self.time_step, output_size=self.output_size
            )
        else:  # 集成过程，self.read_path 为集成器读取的目录，即预测器保存结果的目录
            read_result = ReadResult(
                folder=self.read_path, time_step=self.time_step, output_size=self.output_size, writer=self.writer
            )
            read_result.set_feature(feature=self.parameter_data['Feature'])
            read_result.set_target(target=self.parameter_data['Target'])
            read_result.set_model(model=self.parameter_data['Model'])
            print("成功完成 设置特征、目标和预测结果列名！")
            self.writer.add_text("成功完成 设置特征、目标和预测结果列名", filename="Logs", folder="documents", suffix="log")
            read_result.select_model(
                method=self.parameter_data['ModelSelectionMethod'],
                number=self.parameter_data['ModelSelectionNumber'],
                how=self.parameter_data['ModelSelectionHow']
            )
            print(f"选择模型：{read_result.model_variables}")
            self.writer.add_text(f"选择模型：{read_result.model_variables}",
                                 filename="Logs", folder="documents", suffix="log")
            read_result.convert()
            print("成功完成 预测数据读取！")
            self.writer.add_text("成功完成 预测数据读取！", filename="Logs", folder="documents", suffix="log")
            # 从 read_result 中提取数据
            time_dataset_path = read_result.time_dataset_path
            time_unknown_variables = read_result.time_unknown_variables
            time_known_variables = read_result.time_known_variables
            target_variables = read_result.target_variables
            # 封装成 ReadDataset 类。
            self.dataset = ReadDataset(
                path=time_dataset_path, time_step=self.time_step, output_size=self.output_size, writer=self.writer,
                add_history_features=self.parameter_data['AddHistoryFeatures'], normalization=normalization
            )
            self.dataset.set_feature(time_known=time_known_variables, time_unknown=time_unknown_variables)
            self.dataset.set_target(target=target_variables)
            print("成功完成 特征和目标读取！")
            self.writer.add_text("成功完成 特征和目标读取！", filename="Logs", folder="documents", suffix="log")
            self.dataset.convert()
            print("成功完成 特征转换！")
            self.writer.add_text("成功完成 特征转换！", filename="Logs", folder="documents", suffix="log")
            # 封装成 PackageDataset 类。
            train_start, valid_start, test_start = read_result.train_start, read_result.valid_start, read_result.test_start
            train_end, valid_end, test_end = read_result.train_end, read_result.valid_end, read_result.test_end
            self.package = PackageDataset(
                self.dataset, train_start=train_start, train_end=train_end, valid_start=valid_start,
                valid_end=valid_end, test_start=test_start, test_end=test_end, train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size, sample_gap=self.sample_gap, k_fold=self.k_fold,
                shuffle=self.shuffle, time_step=self.time_step, output_size=self.output_size
            )
        # 读取数据集封装结果，可能来源于 Package
        self.train_target, self.valid_target, self.test_target = self.package.train_target, self.package.valid_target, self.package.test_target
        print("成功完成 数据集封装！")
        self.writer.add_text("成功完成 数据集封装！", filename="Logs", folder="documents", suffix="log")
        # 处理 weights 参数
        if self.parameter_data['Weights'] is not None:
            self.weights = list(self.parameter_data['Weights'])
            self.weights_train = [w/len(self.package.target_variables) for w in self.weights for _ in range(len(self.package.target_variables))]
        self.writer.write_file(self.package, filename="package", folder="data")  # 将数据集写入 data 目录下
        self.package.write_data(writer=self.writer)
        self.parameter.write_parameter(writer=self.writer)
        print("成功完成 数据集和参数写入！")
        self.writer.add_text("成功完成 数据集和参数写入！", filename="Logs", folder="documents", suffix="log", save_mode='a+')

    def ML_fold(self, predictor, parameters: Dict, is_normalization: bool, dtype=np.float32):
        """
        实例化、训练和预测机器学习模型（交叉验证）。
        :param predictor: 机器学习模型类，不需要实例化。
        :param parameters: 机器学习模型参数，dict 类型。
        :param is_normalization: bool 类型，是否标准化。
        :param dtype: 数据类型，默认为 np.float32。
        :return: 每一折验证结果的拼接，最终得到整个泛化后的训练集预测结果（pd.DataFrame）
        """
        fold_predict = np.array([])  # 初始化验证结果
        # 获取数据
        if is_normalization:
            trainer_feature = self.package.train_trainer_norm_feature_fold
            trainer_target = self.package.train_trainer_norm_target_fold
        else:
            trainer_feature = self.package.train_trainer_feature_fold
            trainer_target = self.package.train_trainer_target_fold
        # 执行交叉验证
        for idx, feature_target in enumerate(zip(trainer_feature, trainer_target)):
            features, targets = feature_target
            cprint("\n" + "*" * 105 + f"\n正在执行 {predictor.__name__} 模型第 {idx + 1} 折交叉验证...\n" + "*" * 105, text_color='蓝色')
            self.writer.add_text(f"执行 {predictor.__name__} 模型第 {idx + 1} 折交叉验证。",
                                 filename="Logs", folder="documents", suffix="log")
            model = predictor(**parameters)  # 重置模型
            train_trainer_feature, train_evaler_feature, valid_feature = features
            train_trainer_target, train_evaler_target, valid_target = targets
            model.fit(train_trainer_feature.to_numpy(), train_trainer_target.to_numpy().squeeze())
            valid_predict = model.predict(valid_feature.to_numpy())
            fold_predict = np.concatenate([fold_predict, valid_predict], axis=0) if fold_predict.size else valid_predict
        # 处理预测结果
        fold_predict = pd.DataFrame(fold_predict, columns=self.package.target_columns, index=range(1, len(fold_predict)+1))
        if is_normalization:
            for target_name in self.package.target_variables:
                current_cols = [col for col in self.package.target_columns if col.startswith(target_name)]
                fold_predict[current_cols] = self.package.normalization_target.denorm(
                    fold_predict[current_cols], columns=target_name
                )
        fold_predict = fold_predict.astype(dtype)  # 转换数据类型
        self.writer.add_text(f"完成 {predictor.__name__} 模型交叉验证训练和预测。",
                             filename="Logs", folder="documents", suffix="log", save_mode="a+")
        return fold_predict

    def ML_all(self, predictor, parameters: Dict, is_normalization: bool, dtype=np.float32):
        """
        实例化、训练和预测机器学习模型（在所有数据上训练）。
        :param predictor: 机器学习模型类，不需要实例化。
        :param parameters: 机器学习模型参数，dict 类型。
        :param is_normalization: bool 类型，是否标准化。
        :param dtype: 数据类型，默认为 np.float32。
        :return: 训练后的模型（在所有数据上训练的结果）, 各数据集预测结果（pd.DataFrame），此时训练集的预测结果并不是泛化结果。
        """
        cprint("\n" + "*" * 105 + f"\n正在执行 {predictor.__name__} 模型在所有训练数据上训练和预测...\n" + "*" * 105, text_color='蓝色')
        self.writer.add_text(f"执行 {predictor.__name__} 模型在所有训练数据上训练和预测。",
                             filename="Logs", folder="documents", suffix="log", save_mode="a+")
        model = predictor(**parameters)
        if is_normalization:
            train_trainer_feature = self.package.train_trainer_norm_feature_all
            train_trainer_target = self.package.train_trainer_norm_target_all
            train_feature = self.package.train_feature_norm
            valid_feature = self.package.valid_feature_norm
            test_feature = self.package.test_feature_norm
        else:
            train_trainer_feature = self.package.train_trainer_feature_all
            train_trainer_target = self.package.train_trainer_target_all
            train_feature = self.package.train_feature
            valid_feature = self.package.valid_feature
            test_feature = self.package.test_feature
        # 执行预测任务
        model.fit(train_trainer_feature.to_numpy(), train_trainer_target.to_numpy().squeeze())
        train_predict = model.predict(train_feature.to_numpy())
        valid_predict = model.predict(valid_feature.to_numpy())
        test_predict = model.predict(test_feature.to_numpy())
        # 处理预测结果
        train_predict = pd.DataFrame(train_predict, columns=self.package.target_columns, index=range(1, len(train_predict)+1))
        valid_predict = pd.DataFrame(valid_predict, columns=self.package.target_columns, index=range(1, len(valid_predict)+1))
        test_predict = pd.DataFrame(test_predict, columns=self.package.target_columns, index=range(1, len(test_predict)+1))
        if is_normalization:
            for target_name in self.package.target_variables:
                current_cols = [col for col in self.package.target_columns if col.startswith(target_name)]
                train_predict[current_cols] = self.package.normalization_target.denorm(
                    train_predict[current_cols], columns=target_name
                )
                valid_predict[current_cols] = self.package.normalization_target.denorm(
                    valid_predict[current_cols], columns=target_name
                )
                test_predict[current_cols] = self.package.normalization_target.denorm(
                    test_predict[current_cols], columns=target_name
                )
        # 转换数据类型
        train_predict = train_predict.astype(dtype)
        valid_predict = valid_predict.astype(dtype)
        test_predict = test_predict.astype(dtype)
        return model, (train_predict, valid_predict, test_predict)

    def _DL_parameter(self, predictor, parameters: Dict, criterion, monitor, need_save=True, **kwargs):
        """
        获取深度学习参数。
        :param predictor: 深度学习模型类，不需要实例化。
        :param parameters: 深度学习模型和训练参数，dict 类型。
        :param criterion: 损失函数。
        :param monitor: 监视器。
        :param need_save: 是否保存训练结果。
        :param kwargs: 其他参数（与 parameters 参数合并）。
        :return:
        """
        # 从 predictor_parameters 和 kwargs 筛选出深度学习模型参数 和 深度学习训练参数
        train_known_parameters = (
            'epochs', 'criterion', 'optimizer', 'scheduler', 'monitor', 'clip_norm', 'device', 'best_model_dir',
            'loss_figure_path', 'monitor_figure_path', 'loss_result_path', 'monitor_result_path', 'lr_sec_path',
            'monitor_name', 'loss_title', 'monitor_title', 'loss_yscale', 'monitor_yscale', 'train_back', 'show_draw',
            'ignore_draw_epoch', 'ignore_draw_process'
        )
        optim_known_parameters = (
            'learning_rate', 'weight_decay', 'ReduceLROnPlateau_factor', 'ReduceLROnPlateau_patience',
            'ReduceLROnPlateau_threshold', 'weights'
        )
        parameters.update(kwargs)  # 更新参数。这里的更新原位修改，但初始化模型和训练用的就是这组参数，这是一件好事（会记录在文档里）。
        figure_type = parameters.pop('figure_type')  # 图像保存类型
        model_parameters, optim_parameters, train_parameters = dict(), dict(), dict()
        for key in parameters.keys():
            if key in train_known_parameters:
                train_parameters[key] = parameters[key]
            elif key in optim_known_parameters:
                optim_parameters[key] = parameters[key]
            else:
                model_parameters[key] = parameters[key]
        # 添加模型参数 和 训练参数
        if issubclass(predictor, BaseSequence):  # 序列模型
            model_parameters['input_size'] = self.package.input_size_seq
        elif issubclass(predictor, BaseRegression):  # 回归模型
            model_parameters['input_size'] = self.package.input_size_reg
        model_parameters['output_size'] = len(self.package.target_columns)
        if 'monitor_name' not in train_parameters:
            train_parameters['monitor_name'] = monitor.__name__
        if 'monitor_title' not in train_parameters:
            train_parameters['monitor_title'] = f"{predictor.__name__} 模型 {monitor.__name__} 监视器"
        if 'loss_title' not in train_parameters:
            train_parameters['loss_title'] = f"{predictor.__name__} 模型 {criterion.__name__} 损失值"
        # 生成保存路径参数
        if need_save:
            model_dir = os.path.join(self.save_dir, 'models', predictor.__name__)  # type: ignore
            figure_dir = os.path.join(self.save_dir, 'figures', predictor.__name__)  # type: ignore
            result_dir = os.path.join(self.save_dir, 'results', predictor.__name__)  # type: ignore
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            train_parameters['best_model_dir'] = model_dir
            train_parameters['loss_figure_path'] = os.path.join(figure_dir, predictor.__name__+'损失值曲线.'+figure_type)  # type: ignore
            train_parameters['monitor_figure_path'] = os.path.join(figure_dir, predictor.__name__+'监视值曲线.'+figure_type)  # type: ignore
            train_parameters['loss_result_path'] = os.path.join(result_dir, predictor.__name__+'损失值.csv')  # type: ignore
            train_parameters['monitor_result_path'] = os.path.join(result_dir, predictor.__name__+'监视值.csv')  # type: ignore
            train_parameters['lr_sec_path'] = os.path.join(result_dir, predictor.__name__+'学习率与每秒样本数.csv')  # type: ignore
        return model_parameters, optim_parameters, train_parameters

    def write_criterion_monitor(self, criterion_instance, monitor_instance):
        if hasattr(criterion_instance, 'get_parameters'):  # 将损失函数和损失函数参数写入日志
            criterion_info = f"损失函数为：{criterion_instance.__class__.__name__}；损失函数的参数为：{criterion_instance.get_parameters()}。"
        else:
            criterion_info = f"损失函数为：{criterion_instance.__class__.__name__}。"
        self.writer.add_text(criterion_info, filename="Logs", folder="documents", suffix="log", end='')
        if hasattr(monitor_instance, 'get_parameters'):  # 将监视器和监视器参数写入日志
            monitor_info = f"监视器为：{monitor_instance.__class__.__name__}；监视器的参数为：{monitor_instance.get_parameters()}。"
        else:
            monitor_info = f"监视器为：{monitor_instance.__class__.__name__}。"
        self.writer.add_text(monitor_info, filename="Logs", folder="documents", suffix="log", end='')

    def DL_fold(self, predictor, parameters: Dict, is_normalization:bool, criterion, monitor, dtype=np.float32, **kwargs):
        """
        实例化、训练和预测深度学习模型（交叉验证）。
        :param predictor: 深度学习模型类，不需要实例化。
        :param parameters: 深度学习模型、优化和训练相关参数，dict 类型。
        :param is_normalization: bool 类型，是否标准化。
        :param criterion: 损失函数。
        :param monitor: 模型监视器。
        :param dtype: 数据类型，默认为 np.float32。
        :param kwargs: 其他参数（与 parameters 参数合并），需要包含 weights、figure_type 参数。
        :return: 每一折验证结果的拼接，最终得到整个泛化后的训练集预测结果（pd.DataFrame）
        """
        fold_predict = np.array([])  # 初始化验证结果
        # 更新训练参数 train_parameters（不保存训练结果）
        model_parameters, optim_parameters, train_parameters = self._DL_parameter(
            predictor, parameters, criterion=criterion, monitor=monitor, need_save=False, **kwargs
        )
        train_parameters['show_draw'] = False  # 交叉验证不显示图像
        # 获取数据
        if is_normalization and issubclass(predictor, BaseSequence):  # 序列模型 标准化
            trainer = self.package.train_trainer_norm_seq_fold
        elif is_normalization and issubclass(predictor, BaseRegression):  # 回归模型 标准化
            trainer = self.package.train_trainer_norm_reg_fold
        elif not is_normalization and issubclass(predictor, BaseSequence):  # 序列模型 非标准化
            trainer = self.package.train_trainer_seq_fold
        elif not is_normalization and issubclass(predictor, BaseRegression):  # 回归模型 非标准化
            trainer = self.package.train_trainer_reg_fold
        else:
            error_info = "模型类必须继承自 BaseSequence 或 BaseRegression，否则无法判别任务类型！如果继承自 BaseModel，" + \
                         "请具体化为 BaseSequence 或 BaseRegression，即使需要重写 train_epoch 和 predict 方法，" + \
                         "也可以继承自 BaseSequence 或 BaseRegression 后重写！"
            raise ValueError(error_info)
        # 执行交叉验证
        for idx, trainer_k in enumerate(trainer):
            cprint("\n" + "*"*105 + f"\n正在执行 {predictor.__name__} 模型第 {idx+1} 折交叉验证...\n" + "*"*105, text_color='蓝色')
            self.writer.add_text(f"执行 {predictor.__name__} 模型第 {idx+1} 折交叉验证。",
                                 filename="Logs", folder="documents", suffix="log")
            train_parameters_copy = deepcopy(train_parameters)  # 深拷贝，避免修改原参数
            train_parameters_copy['monitor_title'] += f" (k={idx+1}/{self.k_fold})"
            train_parameters_copy['loss_title'] += f" (k={idx+1}/{self.k_fold})"
            train_trainer_dataloader, train_evaler_dataloader, valid_dataloader = trainer_k
            model = predictor(**model_parameters)  # 重置模型
            criterion_instance = criterion(weights=optim_parameters['weights'])  # 重置损失函数
            monitor_instance = monitor(weights=optim_parameters['weights'])  # 重置监视器
            self.write_criterion_monitor(criterion_instance, monitor_instance)
            optimizer = optim.Adam(model.parameters(), lr=optim_parameters['learning_rate'],
                                   weight_decay=optim_parameters['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=optim_parameters['ReduceLROnPlateau_factor'],
                patience=optim_parameters['ReduceLROnPlateau_patience'],
                threshold=optim_parameters['ReduceLROnPlateau_threshold']
            )
            model.fit(train_trainer=train_trainer_dataloader, train_evaler=train_evaler_dataloader, valid_evaler=valid_dataloader,
                      criterion=criterion_instance, optimizer=optimizer, scheduler=scheduler, monitor=monitor_instance,
                      **train_parameters_copy)
            valid_predict = model.predict(valid_dataloader).cpu().numpy()
            fold_predict = np.concatenate([fold_predict, valid_predict], axis=0) if fold_predict.size else valid_predict
        # 处理预测结果
        fold_predict = pd.DataFrame(fold_predict, columns=self.package.target_columns, index=range(1, len(fold_predict)+1))
        if is_normalization:
            for target_name in self.package.target_variables:
                current_cols = [col for col in self.package.target_columns if col.startswith(target_name)]
                fold_predict[current_cols] = self.package.normalization_target.denorm(
                    fold_predict[current_cols], columns=target_name
                )
        fold_predict = fold_predict.astype(dtype)  # 转换数据类型
        self.writer.add_text(f"完成 {predictor.__name__} 模型交叉验证训练和预测。",
                             filename="Logs", folder="documents", suffix="log", save_mode="a+")
        return fold_predict

    def DL_all(self, predictor, parameters: Dict, is_normalization:bool, criterion, monitor, dtype=np.float32, **kwargs):
        """
        实例化、训练和预测深度学习模型（交叉验证）。
        :param predictor: 深度学习模型类，不需要实例化。
        :param parameters: 深度学习模型、优化和训练相关参数，dict 类型。
        :param is_normalization: bool 类型，是否标准化。
        :param criterion: 损失函数。
        :param monitor: 模型监视器。
        :param dtype: 数据类型，默认为 np.float32。
        :param kwargs: 其他参数（与 parameters 参数合并），需要包含 weights、figure_type 参数。
        :return: 每一折验证结果的拼接，最终得到整个泛化后的训练集预测结果（pd.DataFrame）
        """
        cprint("\n" + "*" * 105 + f"\n正在执行 {predictor.__name__} 模型在所有训练数据上训练和预测...\n" + "*" * 105, text_color='蓝色')
        self.writer.add_text(f"执行 {predictor.__name__} 模型在所有训练数据上训练和预测。",
                             filename="Logs", folder="documents", suffix="log")
        # 更新训练参数 train_parameters（保存训练结果）
        model_parameters, optim_parameters, train_parameters = self._DL_parameter(
            predictor, parameters, criterion=criterion, monitor=monitor, need_save=True, **kwargs
        )
        train_parameters['show_draw'] = True  # 正常训练显示图像
        model = predictor(**model_parameters)  # 重置模型
        criterion_instance = criterion(weights=optim_parameters['weights'])  # 重置损失函数
        monitor_instance = monitor(weights=optim_parameters['weights'])  # 重置监视器
        self.write_criterion_monitor(criterion_instance, monitor_instance)
        self.writer.add_text("", end='', filename="Logs", folder="documents", suffix="log", save_mode='a+')  # 换行
        optimizer = optim.Adam(model.parameters(), lr=optim_parameters['learning_rate'],
                               weight_decay=optim_parameters['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=optim_parameters['ReduceLROnPlateau_factor'],
            patience=optim_parameters['ReduceLROnPlateau_patience'],
            threshold=optim_parameters['ReduceLROnPlateau_threshold']
        )
        # 获取数据
        if is_normalization and issubclass(predictor, BaseSequence):  # 序列模型 标准化
            train_trainer = self.package.train_trainer_norm_seq_all
            train_evaler = self.package.train_evaler_norm_seq
            valid_evaler = self.package.valid_evaler_norm_seq
            test_evaler = self.package.test_evaler_norm_seq
        elif is_normalization and issubclass(predictor, BaseRegression):  # 回归模型 标准化
            train_trainer = self.package.train_trainer_norm_reg_all
            train_evaler = self.package.train_evaler_norm_reg
            valid_evaler = self.package.valid_evaler_norm_reg
            test_evaler = self.package.test_evaler_norm_reg
        elif not is_normalization and issubclass(predictor, BaseSequence):  # 序列模型 非标准化
            train_trainer = self.package.train_trainer_seq_all
            train_evaler = self.package.train_evaler_seq
            valid_evaler = self.package.valid_evaler_seq
            test_evaler = self.package.test_evaler_seq
        elif not is_normalization and issubclass(predictor, BaseRegression):  # 回归模型 非标准化
            train_trainer = self.package.train_trainer_reg_all
            train_evaler = self.package.train_evaler_reg
            valid_evaler = self.package.valid_evaler_reg
            test_evaler = self.package.test_evaler_reg
        else:
            error_info = "模型类必须继承自 BaseSequence 或 BaseRegression，否则无法判别任务类型！如果继承自 BaseModel，" + \
                         "请具体化为 BaseSequence 或 BaseRegression，即使需要重写 train_epoch 和 predict 方法，" + \
                         "也可以继承自 BaseSequence 或 BaseRegression 后重写！"
            raise ValueError(error_info)
        # 执行预测任务
        model.fit(train_trainer=train_trainer, train_evaler=train_evaler, valid_evaler=valid_evaler,
                  criterion=criterion_instance, optimizer=optimizer, scheduler=scheduler, monitor=monitor_instance,
                  **train_parameters)
        train_predict = model.predict(train_evaler).cpu().numpy()
        valid_predict = model.predict(valid_evaler).cpu().numpy()
        test_predict = model.predict(test_evaler).cpu().numpy()
        # 处理预测结果
        train_predict = pd.DataFrame(train_predict, columns=self.package.target_columns, index=range(1, len(train_predict) + 1))
        valid_predict = pd.DataFrame(valid_predict, columns=self.package.target_columns, index=range(1, len(valid_predict) + 1))
        test_predict = pd.DataFrame(test_predict, columns=self.package.target_columns, index=range(1, len(test_predict) + 1))
        if is_normalization:
            for target_name in self.package.target_variables:
                current_cols = [col for col in self.package.target_columns if col.startswith(target_name)]
                train_predict[current_cols] = self.package.normalization_target.denorm(
                    train_predict[current_cols], columns=target_name
                )
                valid_predict[current_cols] = self.package.normalization_target.denorm(
                    valid_predict[current_cols], columns=target_name
                )
                test_predict[current_cols] = self.package.normalization_target.denorm(
                    test_predict[current_cols], columns=target_name
                )
        # 转换数据类型
        train_predict = train_predict.astype(dtype)
        valid_predict = valid_predict.astype(dtype)
        test_predict = test_predict.astype(dtype)
        return model, (train_predict, valid_predict, test_predict)

    def model(self, predictor, *, is_normalization=True, save_result=True, save_figure=True,
              show_result=True, show_figure=False, criterion=None, monitor=None, dtype=np.float32):
        """
        训练和预测一个模型，并保存结果和评估指标。
        :param predictor: 模型类，不需要实例化。需要有 fit 和 predict 方法。
        :param is_normalization: 是否标准化，默认为 True。
        :param save_result: 是否保存预测结果、预测指标、日志信息，默认为 True。
        :param save_figure: 是否保存预测结果走势图，默认为 True。
        :param show_result: 是否在控制台打印预测指标信息，默认为 True。
        :param show_figure: 是否展示绘制的预测结果走势图，默认为 False。
        :param criterion: 深度学习损失函数类，无需实例化，深度学习必须参数。
        :param monitor: 深度学习模型监视器类，无需实例化，深度学习必须参数。
        :param dtype: 预测结果数据类型，默认为 np.float32。
        Note: 如果是深度学习模型，则需要传入 criterion、monitor 参数。
        :return: 训练后的模型。
        """
        start_time = time.perf_counter()
        predictor_name = predictor.__name__  # 模型名称
        if show_result:
            cprint(f"开始训练和预测 {predictor_name} 模型...", text_color="白色", end='\n')
        self.writer.add_text(f"开始训练和预测 {predictor_name} 模型。", filename="Logs", folder="documents", suffix="log", save_mode="a+")
        # 处理参数
        all_parameters = dict()
        all_parameters.update(self.parameter_model[predictor_name])  # 添加模型参数

        # 模型实例化、训练、预测
        if issubclass(predictor, base.BaseEstimator):  # 机器学习模型
            fold_predict = self.ML_fold(predictor, all_parameters, is_normalization=is_normalization, dtype=dtype)
            trained_model, predict_values = self.ML_all(predictor, all_parameters, is_normalization=is_normalization, dtype=dtype)
            if save_result:
                self.writer.write_file(trained_model, filename=predictor_name, folder='models')
        elif issubclass(predictor, nn.Module):
            all_parameters.update(self.parameter_model['Train'])  # 添加训练参数
            fold_predict = self.DL_fold(
                predictor, all_parameters, is_normalization=is_normalization, criterion=criterion, monitor=monitor,
                dtype=dtype, weights=self.weights_train, figure_type=self.parameter_data['FigureType']
            )
            trained_model, predict_values = self.DL_all(
                predictor, all_parameters, is_normalization=is_normalization, criterion=criterion, monitor=monitor,
                dtype=dtype, weights=self.weights_train, figure_type=self.parameter_data['FigureType']
            )

        else:
            raise ValueError("模型类必须是 sklearn.base.BaseEstimator 或 torch.nn.Module 的子类！")
        train_predict, valid_predict, test_predict = predict_values
        # 处理预测结果、评估指标、绘制图像、展示结果、保存结果、展示图像、保存图像
        self.metrics_show_save(
            predictor_name=predictor_name, fold_predict=fold_predict, train_predict=train_predict,
            valid_predict=valid_predict, test_predict=test_predict, save_result=save_result,
            save_figure=save_figure, show_result=show_result, show_figure=show_figure, dtype=dtype
        )
        # 保存模型参数
        self.writer.add_param(param_desc=f"{predictor_name} 模型参数", param_dict=self.parameter_model[predictor_name],
                              filename="predictor parameters", folder='documents')
        # 记录运行时间
        end_time = time.perf_counter()
        cprint(f"{predictor_name} 模型训练和预测结束，用时 {end_time - start_time} 秒。", text_color="白色")
        self.writer.add_text(f"{predictor_name} 模型训练和预测结束，用时 {(end_time - start_time):.2f} 秒。",
                             filename="Logs", folder="documents", suffix="log", save_mode='a+')
        # 返回训练后的模型
        return trained_model

    def metrics_show_save(self, predictor_name, fold_predict, train_predict, valid_predict, test_predict,
                          save_result, save_figure, show_result, show_figure, dtype):
        """根据提供的信息进行：处理预测结果、评估指标、绘制图像、展示结果、保存结果、展示图像、保存图像"""
        # 添加上下界限制
        lower_bound, upper_bound = self.parameter_data['LowerBound'], self.parameter_data['UpperBound']
        lower_bound = dtype(lower_bound) if lower_bound is not None else None
        upper_bound = dtype(upper_bound) if upper_bound is not None else None
        if lower_bound is not None:
            fold_predict[fold_predict < lower_bound] = lower_bound
            train_predict[train_predict < lower_bound] = lower_bound
            valid_predict[valid_predict < lower_bound] = lower_bound
            test_predict[test_predict < lower_bound] = lower_bound
        if upper_bound is not None:
            fold_predict[fold_predict > upper_bound] = upper_bound
            train_predict[train_predict > upper_bound] = upper_bound
            valid_predict[valid_predict > upper_bound] = upper_bound
            test_predict[test_predict > upper_bound] = upper_bound
        # 计算每个目标的评估指标（key 为目标名，value 该目标的指标结果（包含预测时间步，pd.DataFrame））
        fold_metrics_sub, train_metrics_sub, valid_metrics_sub, test_metrics_sub = [], [], [], []
        for col_name in self.package.target_columns:
            target_name, target_step = col_name.rsplit('_-', 1)
            fold_metrics_temp = calculate_metrics(self.train_target[[col_name]], fold_predict[[col_name]])
            train_metrics_temp = calculate_metrics(self.train_target[[col_name]], train_predict[[col_name]])
            valid_metrics_temp = calculate_metrics(self.valid_target[[col_name]], valid_predict[[col_name]])
            test_metrics_temp = calculate_metrics(self.test_target[[col_name]], test_predict[[col_name]])
            fold_metrics_sub.append({'target': target_name, 'model': predictor_name, 'step': target_step, **fold_metrics_temp})
            train_metrics_sub.append({'target': target_name, 'model': predictor_name, 'step': target_step, **train_metrics_temp})
            valid_metrics_sub.append({'target': target_name, 'model': predictor_name, 'step': target_step, **valid_metrics_temp})
            test_metrics_sub.append({'target': target_name, 'model': predictor_name, 'step': target_step, **test_metrics_temp})
        fold_metrics_sub = pd.DataFrame(fold_metrics_sub, index=range(len(fold_metrics_sub)))
        train_metrics_sub = pd.DataFrame(train_metrics_sub, index=range(len(train_metrics_sub)))
        valid_metrics_sub = pd.DataFrame(valid_metrics_sub, index=range(len(valid_metrics_sub)))
        test_metrics_sub = pd.DataFrame(test_metrics_sub, index=range(len(test_metrics_sub)))
        # 计算整体评估指标
        fold_metrics, train_metrics, valid_metrics, test_metrics = [], [], [], []
        for target_name in self.package.target_variables:
            col_name = [c for c in self.package.target_columns if re.match(f"^{target_name}_-\d+", c)]
            fold_metrics_temp = calculate_metrics(self.train_target[col_name], fold_predict[col_name], weights=self.weights)
            train_metrics_temp = calculate_metrics(self.train_target[col_name], train_predict[col_name], weights=self.weights)
            valid_metrics_temp = calculate_metrics(self.valid_target[col_name], valid_predict[col_name], weights=self.weights)
            test_metrics_temp = calculate_metrics(self.test_target[col_name], test_predict[col_name], weights=self.weights)
            fold_metrics.append({'target': target_name, 'model': predictor_name, **fold_metrics_temp})
            train_metrics.append({'target': target_name, 'model': predictor_name, **train_metrics_temp})
            valid_metrics.append({'target': target_name, 'model': predictor_name, **valid_metrics_temp})
            test_metrics.append({'target': target_name, 'model': predictor_name, **test_metrics_temp})
        fold_metrics = pd.DataFrame(fold_metrics, index=range(len(fold_metrics)))
        train_metrics = pd.DataFrame(train_metrics, index=range(len(train_metrics)))
        valid_metrics = pd.DataFrame(valid_metrics, index=range(len(valid_metrics)))
        test_metrics = pd.DataFrame(test_metrics, index=range(len(test_metrics)))
        # 绘制和保存预测图像（分目标）
        if show_figure or save_figure:
            for target_name in self.package.target_variables:
                # 获取每个目标变量的真实值和目标值
                current_cols = [col for col in self.package.target_columns if col.startswith(target_name)]
                current_fold_predict = fold_predict[current_cols]
                current_train_target, current_train_predict = self.train_target[current_cols], train_predict[current_cols]
                current_valid_target, current_valid_predict = self.valid_target[current_cols], valid_predict[current_cols]
                current_test_target, current_test_predict = self.test_target[current_cols], test_predict[current_cols]
                # 保存交叉验证图片
                for i in range(current_train_target.shape[-1]):
                    title = f"{predictor_name} 交叉验证 {target_name} 目标 {i + 1} 步预测"
                    filename = title if save_figure else None
                    self.writer.draw(
                        [current_train_target.iloc[:, i].to_numpy(), current_fold_predict.iloc[:, i].to_numpy()],
                        title=title, legend=[f'true_{i + 1}', f"{predictor_name}_{i + 1}"], filename=filename,
                        folder='figures', suffix=self.parameter_data['FigureType'], show=show_figure
                    )
                # 保存训练集图片
                for i in range(current_train_target.shape[-1]):
                    title = f"{predictor_name} 训练集 {target_name} 目标 {i + 1} 步预测"
                    filename = title if save_figure else None
                    self.writer.draw(
                        [current_train_target.iloc[:, i].to_numpy(), current_train_predict.iloc[:, i].to_numpy()],
                        title=title, legend=[f'true_{i + 1}', f"{predictor_name}_{i + 1}"], filename=filename,
                        folder='figures', suffix=self.parameter_data['FigureType'], show=show_figure
                    )
                # 保存验证集图片
                for i in range(current_valid_target.shape[-1]):
                    title = f"{predictor_name} 验证集 {target_name} 目标 {i + 1} 步预测"
                    filename = title if save_figure else None
                    self.writer.draw(
                        [current_valid_target.iloc[:, i].to_numpy(), current_valid_predict.iloc[:, i].to_numpy()],
                        title=title, legend=[f'true_{i + 1}', f"{predictor_name}_{i + 1}"], filename=filename,
                        folder='figures', suffix=self.parameter_data['FigureType'], show=show_figure
                    )
                # 保存测试集图片
                for i in range(current_test_target.shape[-1]):
                    title = f"{predictor_name} 测试集 {target_name} 目标 {i + 1} 步预测"
                    filename = title if save_figure else None
                    self.writer.draw(
                        [current_test_target.iloc[:, i].to_numpy(), current_test_predict.iloc[:, i].to_numpy()],
                        title=title, legend=[f'true_{i + 1}', f"{predictor_name}_{i + 1}"], filename=filename,
                        folder='figures', suffix=self.parameter_data['FigureType'], show=show_figure
                    )
        # 打印和保存结果
        if show_result:
            cprint(f"{predictor_name} 模型交叉验证评估指标：\n {str(fold_metrics)}", text_color="青色", end='\n\n')
            cprint(f"{predictor_name} 模型训练集评估指标：\n {str(train_metrics)}", text_color="蓝色", end='\n\n')
            cprint(f"{predictor_name} 模型验证集评估指标：\n {str(valid_metrics)}", text_color="紫色", end='\n\n')
            cprint(f"{predictor_name} 模型测试集评估指标：\n {str(test_metrics)}", text_color="黄色", end='\n\n')
        if save_result:
            # 保存预测结果
            self.package.write_predict(writer=self.writer, value=fold_predict, model=predictor_name, dataset='fold')
            self.package.write_predict(writer=self.writer, value=train_predict, model=predictor_name, dataset='train')
            self.package.write_predict(writer=self.writer, value=valid_predict, model=predictor_name, dataset='valid')
            self.package.write_predict(writer=self.writer, value=test_predict, model=predictor_name, dataset='test')
            # 保存评估指标
            self.writer.add_df(fold_metrics, axis=0, filename="fold metrics", folder="results", suffix="xlsx",
                               reset_index=True, reset_drop=True, sort_list=['target', 'model'], sort_ascending=True)
            self.writer.add_df(train_metrics, axis=0, filename="train metrics", folder="results", suffix="xlsx",
                               reset_index=True, reset_drop=True, sort_list=['target', 'model'], sort_ascending=True)
            self.writer.add_df(valid_metrics, axis=0, filename="valid metrics", folder="results", suffix="xlsx",
                               reset_index=True, reset_drop=True, sort_list=['target', 'model'], sort_ascending=True)
            self.writer.add_df(test_metrics, axis=0, filename="test metrics", folder="results", suffix="xlsx",
                               reset_index=True, reset_drop=True, sort_list=['target', 'model'], sort_ascending=True)
            self.writer.add_df(fold_metrics_sub, axis=0, filename="fold metrics (detailed)", folder="results", suffix="xlsx",
                               reset_index=True, reset_drop=True, sort_list=['target', 'model', 'step'], sort_ascending=True)
            self.writer.add_df(train_metrics_sub, axis=0, filename="train metrics (detailed)", folder="results", suffix="xlsx",
                               reset_index=True, reset_drop=True, sort_list=['target', 'model', 'step'], sort_ascending=True)
            self.writer.add_df(valid_metrics_sub, axis=0, filename="valid metrics (detailed)", folder="results", suffix="xlsx",
                               reset_index=True, reset_drop=True, sort_list=['target', 'model', 'step'], sort_ascending=True)
            self.writer.add_df(test_metrics_sub, axis=0, filename="test metrics (detailed)", folder="results", suffix="xlsx",
                               reset_index=True, reset_drop=True, sort_list=['target', 'model', 'step'], sort_ascending=True)

    def all_models(self, predictors: Union[List,Tuple], *, is_normalization:Union[List,Tuple,bool]=True, save_result=True,
                   save_figure=True, show_result=True, show_figure=False, criterion=None, monitor=None, dtype=np.float32) -> None:
        """
        训练和预测多个模型，并保存结果和评估指标。
        :param predictors: 由模型类构成的 list 或 tuple，不需要实例化。
        :param is_normalization: 是否标准化。可以接受 bool 类型、list 类型或 tuple 类型。如果为 bool 类型，则所有模型都使用相同的标准化方式；
                              如果为 list 类型或 tuple 类型，则每个模型使用对应的标准化方式（保证长度与 models_trained 长度一致）。
        :param save_result: 是否保存结果，默认为 True。
        :param save_figure: 是否保存图像，默认为 True。
        :param show_result: 是否显示结果，默认为 True。
        :param show_figure: 是否显示图像，默认为 False。
        :param criterion: 深度学习损失函数类，无需实例化，深度学习必须参数。
        :param monitor: 深度学习模型监视器类，无需实例化，深度学习必须参数。
        :param dtype: 预测结果数据类型，默认为 np.float32。
        :return: None
        """
        if isinstance(is_normalization, bool):
            is_normalization = [is_normalization] * len(predictors)
        for m, is_norm in zip(predictors, is_normalization):
            try:  # 运行时发生错误，忽略该模型，继续运行
                self.model(
                    m, is_normalization=is_norm, save_result=save_result, save_figure=save_figure,
                    show_result=show_result, show_figure=show_figure, criterion=criterion, monitor=monitor, dtype=dtype
                )
            except BaseException:
                error_message = traceback.format_exc()
                self.error_number += 1  # 错误数量加 1
                self.writer.add_text(
                    "", filename="Logs", folder="documents", suffix="log", save_mode='a+'
                )
                self.writer.add_text(
                    f"Error: {error_message}", filename="Logs", folder="documents", suffix="log", save_mode='a+'
                )
                self.writer.write(self.save_mode)  # 保存所有暂存结果，防止丢失
                info = f'\n{m.__name__} 训练时发生错误，错误信息已经保存到日志，请到日志中查看！已忽略该模型的训练，开始训练下一个模型！\n'
                cprint("-" * 100 + info + '-' * 100 + '\n', text_color='红色', background_color='浅黄色', style='加粗')
                continue

    def persistence(self, save_result=True, save_figure=True, show_result=True, show_figure=False, dtype=np.float32):
        """
        运行 Persistence 模型，并保存结果和评估指标。
        :param save_result: 是否保存预测结果、预测指标、日志信息，默认为 True。
        :param save_figure: 是否保存预测结果走势图，默认为 True。
        :param show_result: 是否在控制台打印预测指标信息，默认为 True。
        :param show_figure: 是否展示绘制的预测结果走势图，默认为 False。
        :param dtype: 预测结果数据类型，默认为 np.float32。
        :return: None
        """
        start_time = time.perf_counter()
        predictor_name = 'Persistence'  # 模型名称
        if show_result:
            cprint(f"开始执行 {predictor_name} 模型...", text_color="白色", end='\n')
        self.writer.add_text(f"开始执行 {predictor_name} 模型。", filename="Logs", folder="documents", suffix="log", save_mode="a+")
        # 获取 Persistence 结果（fold 的 Persistence 结果和 train 的 Persistence 结果一致）
        feature_cols = [f"{var}_1" for var in self.package.target_variables]
        target_cols = [f"{var}_{step}" for step in range(-1, -self.output_size, -1) for var in self.package.target_variables]
        self.train_persistence = pd.concat(
            [self.package.train_feature[feature_cols], self.package.train_target[target_cols]], axis=1
        )
        self.valid_persistence = pd.concat(
            [self.package.valid_feature[feature_cols], self.package.valid_target[target_cols]], axis=1
        )
        self.test_persistence = pd.concat(
            [self.package.test_feature[feature_cols], self.package.test_target[target_cols]], axis=1
        )
        train_predict = pd.DataFrame(self.train_persistence.values, columns=self.package.target_columns, index=range(1, len(self.train_persistence)+1))
        valid_predict = pd.DataFrame(self.valid_persistence.values, columns=self.package.target_columns, index=range(1, len(self.valid_persistence)+1))
        test_predict = pd.DataFrame(self.test_persistence.values, columns=self.package.target_columns, index=range(1, len(self.test_persistence)+1))
        fold_predict = train_predict
        # 处理预测结果、评估指标、绘制图像、展示结果、保存结果、展示图像、保存图像
        self.metrics_show_save(
            predictor_name=predictor_name, fold_predict=fold_predict, train_predict=train_predict,
            valid_predict=valid_predict, test_predict=test_predict, save_result=save_result,
            save_figure=save_figure, show_result=show_result, show_figure=show_figure, dtype=dtype
        )
        # 记录运行时间
        end_time = time.perf_counter()
        cprint(f"{predictor_name} 模型执行结束，用时 {end_time - start_time} 秒。", text_color="白色")
        self.writer.add_text(f"{predictor_name} 模型执行结束，用时 {(end_time - start_time):.2f} 秒。",
                             filename="Logs", folder="documents", suffix="log", save_mode='a+')

    def average(self, save_result=True, save_figure=True, show_result=True, show_figure=False, dtype=np.float32):
        """
        运行 Persistence 模型，并保存结果和评估指标。
        :param save_result: 是否保存预测结果、预测指标、日志信息，默认为 True。
        :param save_figure: 是否保存预测结果走势图，默认为 True。
        :param show_result: 是否在控制台打印预测指标信息，默认为 True。
        :param show_figure: 是否展示绘制的预测结果走势图，默认为 False。
        :param dtype: 预测结果数据类型，默认为 np.float32。
        :return: None
        """
        start_time = time.perf_counter()
        predictor_name = 'Average'  # 模型名称
        if show_result:
            cprint(f"开始执行 {predictor_name} 模型...", text_color="白色", end='\n')
        self.writer.add_text(f"开始执行 {predictor_name} 模型。", filename="Logs", folder="documents", suffix="log", save_mode="a+")
        # 获取 Average 结果（fold 的 Average 结果和 train 的 Average 结果一致；集成的数据采用 fold 数据，该数据为泛化数据）
        train_predict = pd.DataFrame()
        valid_predict = pd.DataFrame()
        test_predict = pd.DataFrame()
        pattern_cols = re.compile(r'\S+(\S+)-\d+_-\d+')
        self.average_columns = [var for var in self.package.train_feature.columns if re.match(pattern_cols, var)]
        for var in self.package.target_variables:
            for step in range(1, self.output_size+1):
                cols = [col for col in self.average_columns if re.match(f"{var}(\S+)-\d+_-{step}", col)]
                train_predict[f'{var}_-{step}'] = self.package.train_feature[cols].mean(axis=1).values
                valid_predict[f'{var}_-{step}'] = self.package.valid_feature[cols].mean(axis=1).values
                test_predict[f'{var}_-{step}'] = self.package.test_feature[cols].mean(axis=1).values
        # 更换列的顺序
        train_predict = train_predict[self.package.target_columns]
        valid_predict = valid_predict[self.package.target_columns]
        test_predict = test_predict[self.package.target_columns]
        fold_predict = train_predict
        # 处理预测结果、评估指标、绘制图像、展示结果、保存结果、展示图像、保存图像
        self.metrics_show_save(
            predictor_name=predictor_name, fold_predict=fold_predict, train_predict=train_predict,
            valid_predict=valid_predict, test_predict=test_predict, save_result=save_result,
            save_figure=save_figure, show_result=show_result, show_figure=show_figure, dtype=dtype
        )
        # 记录运行时间
        end_time = time.perf_counter()
        cprint(f"{predictor_name} 模型执行结束，用时 {end_time - start_time} 秒。", text_color="白色")
        self.writer.add_text(f"{predictor_name} 模型执行结束，用时 {(end_time - start_time):.2f} 秒。",
                             filename="Logs", folder="documents", suffix="log", save_mode='a+')

    def run(self, predictors: Union[type,List,Tuple], *, is_normalization:Union[bool,List,Tuple]=True,
            add_persistence=True, add_average=True, save_result=True, save_figure=True, show_result=True,
            show_figure=False, criterion=None, monitor=None, dtype=np.float32):
        """
        训练和预测一个或者多个模型，并保存结果和评估指标。
        :param predictors: 传入一个模型类，或者是由模型类构成的 list 或 tuple，不需要实例化。
        :param is_normalization: 是否标准化。可以接受 bool 类型、list 类型或 tuple 类型。如果为 bool 类型，则所有模型都使用相同的标准化方式；
                              如果为 list 类型或 tuple 类型，则每个模型使用对应的标准化方式（保证长度与 models_trained 长度一致）。
        :param add_persistence: 是否添加 Persistence 模型，默认为 True。
        :param add_average: 是否添加 Average 集成模型，默认为 True。仅在 self.parameter_data['EnsembleMethod'] 不为 null 时生效。
        :param save_result: 是否保存结果，默认为 True。
        :param save_figure: 是否保存图像，默认为 True。
        :param show_result: 是否显示结果，默认为 True。
        :param show_figure: 是否显示图像，默认为 False。
        :param criterion: 深度学习损失函数类，无需实例化，深度学习必须参数。
        :param monitor: 深度学习模型监视器类，无需实例化，深度学习必须参数。
        :param dtype: 预测结果数据类型，默认为 np.float32。
        :return: 由训练后模型构成的列表。
        """
        cache_dir = os.path.join(os.getcwd(), '.cache predictor')
        # 删除缓存目录
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            self.writer.add_text(
                f"缓存目录 {cache_dir} 删除成功！", filename="Logs", folder="documents", suffix="log", save_mode='a+'
            )
        try:
            # 运行 Persistence 模型和 Average 集成模型
            if add_persistence:
                self.persistence()
            if self.parameter_data['EnsembleMethod'] is not None and add_average:
                self.average()
            # 运行用户自定义模型
            if isinstance(predictors, (list, tuple)):
                self.all_models(
                    predictors, is_normalization=is_normalization, save_result=save_result, save_figure=save_figure,
                    show_result=show_result, show_figure=show_figure, criterion=criterion, monitor=monitor, dtype=dtype
                )
            elif isinstance(predictors, type):
                self.model(
                    predictors, is_normalization=is_normalization, save_result=save_result, save_figure=save_figure,
                    show_result=show_result, show_figure=show_figure, criterion=criterion, monitor=monitor, dtype=dtype
                )
            else:
                raise ValueError("predictors 参数必须是模型类，或者是由模型类构成的 list 或 tuple！")
            # 删除缓存目录
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                self.writer.add_text(
                    f"缓存目录 {cache_dir} 删除成功！", filename="Logs", folder="documents", suffix="log", save_mode='a+'
                )
            # 保存暂存结果
            self.writer.write(self.save_mode)
            if self.error_number:
                info = f'所有训练任务结束，系统运行错误次数为：{self.error_number}。错误信息已经保存到日志，请到日志中查看！'
            else:
                info = f'所有训练任务结束，系统运行时未发生错误！'
            cprint("-" * 100 + f"\n{info}\n" + '-' * 100 + '\n', text_color='红色', background_color='青色', style='加粗')
            self.writer.add_text(info, filename="Logs", folder="documents", suffix="log", save_mode="a+")

        except BaseException as e:
            error_message = traceback.format_exc()
            self.writer.add_text(
                "", filename="Logs", folder="documents", suffix="log", save_mode='a+'
            )
            self.writer.add_text(
                f"Error: {error_message}", filename="Logs", folder="documents", suffix="log", save_mode='a+'
            )
            self.writer.write(self.save_mode)
            info = f'系统运行时发生无法解决的错误，错误信息已经保存到日志，请到日志中查看！'
            cprint("-" * 100 + f"\n{info}\n" + '-' * 100 + '\n', text_color='红色', background_color='浅黄色', style='加粗')
            self.writer.add_text(info, filename="Logs", folder="documents", suffix="log", save_mode="a+")
            # 删除缓存目录
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                self.writer.add_text(
                    f"缓存目录 {cache_dir} 删除成功！", filename="Logs", folder="documents", suffix="log", save_mode='a+'
                )
            raise e
