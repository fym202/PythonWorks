# -*- coding: utf-8 -*-
"""
Created on 2023-02-01 16:22

@author: Fan yi ming

Func: 误差分析
"""
import numpy as np


def error_analyze(data_pred, data_true):
    if len(data_pred) != len(data_true):
        print("误差分析：数据格式不一致")
    data_len = len(data_pred)
    # 计算绝对误差
    abs_error = data_pred - data_true
    # 相对误差
    opp_error = abs((data_pred - data_true) / data_true) * 100
    return abs_error, opp_error


data_pred = np.array([7.42, 6.50, 6.61, 6.75])
data_true = np.array([7.74, 6.61, 6.81, 6.97])
abs_error, opp_error = error_analyze(data_pred, data_true)
print(abs_error, opp_error)
