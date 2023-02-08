# -*- coding: utf-8 -*-
"""
Created on 2022-11-30 16:37

@author: Fan yi ming

Func: 实现决策树预测威斯康辛州乳腺癌数据
"""
from sklearn import datasets
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 加载数据
cancer = datasets.load_breast_cancer()
# print(cancer.data)
# print(cancer.target)
# print(cancer.target_names)
# print(cancer.feature_names)
x = cancer.data
y = cancer.target
print(x)
print(y)
# 数据划分和预测
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(xtrain, ytrain)
score = clf.score(xtest, ytest)
ypre = clf.predict(xtest)
print(score)

# 绘图AUC
y_test_proba = clf.predict_proba(xtest)
false_positive_rate, recall, thresholds = roc_curve(ytest, y_test_proba[:, 1])
roc_auc = auc(false_positive_rate, recall)

plt.plot(false_positive_rate, recall, color='blue', label='AUC_orig=%0.3f' % roc_auc)
plt.legend(loc='best', fontsize=15, frameon=False)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()
