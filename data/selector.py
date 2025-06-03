# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/5/2 16:49
# @Author   : 张浩
# @FileName : selector.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_regression, chi2, r_regression

from utils.metrics import *


class metric_best:
    def __init__(self, metric:str, number:int, optimization:str='min'):
        """
        通过传入的指标 metric 和需要筛选出的特征个数 number，选出指标 metric 最好的 number 个特征。
        :param metric: 可选值为 'MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'MSLE', 'RMSLE', 'MedAE', 'MaxError', 'RAE',
                                'RSE', 'Huber', 'R2'（需要是 metrics.metrics 定义的函数）
        :param number: 特征选择后特征的数量。
        :param optimization: 优化方向，'min' 表示最小化，'max' 表示最大化。
        """
        self.metric = metric
        self.number = number
        self.optimization = optimization

    def fit_transform(self, feature:pd.DataFrame, target:pd.Series):
        """
        根据 feature、target 和 weights 计算指标，并选出指标最好的 number 个特征。
        :param feature: 输入的特征，其中每个特征应当为子预测器的模型预测值，否则计算特征与真实值的指标无任何意义！
        :param target: 真实值。
        :return: 特征选择结果，类型为 DataFrame。
        """
        metrics_result = dict()
        target_nd = convert_to_numpy(target.to_frame())  # pd.Series -> pd.DataFrame -> np.ndarray
        assert target_nd.shape[0] == feature.shape[0], f"真实值长度为 {target_nd.shape[0]} 和预测值长度为 {feature.shape[0]} 不匹配！"
        for col in feature.columns:
            feature_nd = convert_to_numpy(feature[[col]])  # pd.DataFrame -> np.ndarray
            metrics_result[col] = eval(self.metric)(true_value=target_nd, predict_value=feature_nd, weights=[1.0])  # 只有一列
        # 选择出来的列名
        if self.optimization == 'min':
            selected_columns = [k for k, v in sorted(metrics_result.items(), key=lambda item: item[1])[:self.number]]
        elif self.optimization == 'max':
            selected_columns = [k for k, v in sorted(metrics_result.items(), key=lambda item: item[1], reverse=True)[:self.number]]
        else:
            raise ValueError("optimization 参数错误！请选择 'min' 或 'max'。")
        return feature[selected_columns]


class Selector:
    def __init__(self, method:str, number:int):
        """
        使用 sklearn.feature_selection.SelectKBest 选择特征。
        :param method: 特征选择方法。可选值为 None、'互信息'、'F检验'、'卡方检验'、'相关系数'。
                        在集成部分，可以选择 'MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'MSLE', 'RMSLE', 'MedAE', 'MaxError',
                                            'RAE', 'RSE', 'Huber', 'R2'。
        :param number: 特征选择后特征的数量。
        :return: 特征选择结果，类型为 DataFrame。
        """
        # 特征选择方法
        self.method = method
        if method is None:
            self.selector_method = None
        elif method == '互信息':
            self.selector_method = SelectKBest(score_func=mutual_info_regression, k=number)
        elif method == 'F检验':
            self.selector_method = SelectKBest(score_func=f_regression, k=number)
        elif method == '卡方检验':
            self.selector_method = SelectKBest(score_func=chi2, k=number)
        elif method == '相关系数':
            self.selector_method = SelectKBest(score_func=r_regression, k=number)
        elif method in ['MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'MSLE', 'RMSLE', 'MedAE', 'MaxError', 'RAE', 'RSE', 'Huber']:
            self.selector_metric = metric_best(metric=method, number=number, optimization='min')
        elif method in ['R2']:
            self.selector_metric = metric_best(metric=method, number=number, optimization='max')
        else:
            raise ValueError("method 参数错误！")

    def fit_transform(self, feature: pd.DataFrame, target: pd.Series):
        if self.method is None:
            return feature
        elif self.method in ['互信息', 'F检验', '卡方检验', '相关系数']:
            feature_selected = self.selector_method.fit_transform(feature, target)  # feature_selected 是 ndarray 类型
            # 转为 DataFrame 类型
            feature_selected = pd.DataFrame(feature_selected, columns=feature.columns[self.selector_method.get_support()])
            return feature_selected
        elif self.method in ['MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'MSLE', 'RMSLE', 'MedAE', 'MaxError', 'RAE', 'RSE', 'Huber', 'R2']:
            feature_selected = self.selector_metric.fit_transform(feature, target)
            return feature_selected
        else:
            raise ValueError("method 参数错误！")
