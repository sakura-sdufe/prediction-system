# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/30 09:32
# @Author   : 张浩
# @FileName : __init__.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from .mlp import MLP
from .cnn import C2L
from .attention import CAttn, CAttnProj


ml_model = ['SVR', 'Ridge', 'RandomForestRegressor', 'GradientBoostingRegressor', 'AdaBoostRegressor', 'BaggingRegressor']
dl_model = ['MLP', 'CAttn', 'CAttnProj', 'C2L']

__all__ = ml_model + dl_model
