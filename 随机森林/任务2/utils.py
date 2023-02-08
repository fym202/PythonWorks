# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
# 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
def getFeatureImportance():
    # 根据如下排序，设置特征重要性
    # 重要程度从大到小排：
    # 1、单位节能 2、单位节水 3、单位节材 4、健康效益 5、单位节地
    # 6、CO2减排效益7、提高就业率  8、单位增长成本  9、总建筑面积  10、节省排污、设施费
    # y_ticks = ['总建筑面积','单位增长成本','单位节能','单位节材','单位节水','单位节地','CO2减排效益','健康效益','提高就业率效益',
    #              '节省排污、设施费']
    feature_importance = np.array([28, 32, 50, 44, 45, 41, 38, 42, 36, 25])
    feature_importance = feature_importance/np.sum(feature_importance)
    return feature_importance

def plot_distrbute(y_train, y_train_pred, y_test, y_test_pred):
    plt.scatter(y_train, y_train_pred, c="blue", label="训练集")
    plt.scatter(y_test, y_test_pred, c="red", label="测试集")
    x = np.linspace(0, 18, 100)
    plt.plot(x,x,c='black')
    plt.xlim([0,18])
    plt.ylim([0,12])
    plt.xlabel("真实值",fontsize=18)
    plt.ylabel("预测值",fontsize=18)
    plt.legend()
    plt.show()