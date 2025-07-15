# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  : Predictor
# @Time     : 2025/4/29 15:27
# @Author   : 张浩
# @FileName : accumulator.py
# @Software : PyCharm
# @Function : 
-------------------------------------------------
"""

class Accumulator:
    def __init__(self, n=1):
        # 初始化变量个数
        self.data = [0.0] * n

    def add(self, *args):
        # 累加函数。注意，这里的累加直接把数放进来就可以，不需要用list或者tuple进行封装
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        # 重置函数。将所有值全部重置为0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        # 索引方法。实例对象obj可以使用方括号索引，并且返回指定值。obj[idx] = 累加器第idx值
        return self.data[idx]
