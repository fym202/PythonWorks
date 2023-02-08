import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt    #画图
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from  sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#定义相关变量
random_1 =35# 样本集选取随机种子
rate_1 = 0.25# 样本集验证集
f_name= "新数据3.csv"
#读入数据
data =pd.read_csv('新数据3.csv',encoding='gb2312')
data_x=data[['单位增长成本','单位节能量','单位节材','单位节水','单位节地','生态环境 废气减排', '  2居民健康','社会     提高就业率','      居民幸福感指数']]


data_y=data['效益 一星级[40,60] ；二星级 [60,80]； 三星级[80,100]']

#'生态环境 废气减排','  2居民健康','社会     提高就业率','      居民幸福感指数'
#分割训练集（后面的会乘分配率）（train_test_split是分配模块函数）
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=rate_1, random_state=random_1)

#将用到的数据集转变成数组的格式
train_x=np.array(train_x)
train_y=np.array(train_y)

test_x=np.array(test_x)
test_y=np.array(test_y)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

#对标签的处理
train_y = np.reshape(train_y, -1)
test_y = np.reshape(test_y, (-1,1))

print(train_y.shape)
print(test_y.shape)


#进行数据的归一化,得到归一标准化后的值

mm_X = StandardScaler()
mm_X.fit(data_x)
train_x = mm_X.transform(train_x)
test_x = mm_X.transform(test_x)

##随机森林分类器

rf=RandomForestRegressor(n_estimators=35,max_depth=None,min_samples_leaf=2,min_samples_split=7,max_features=None,
random_state=0)


print(rf.fit(train_x, train_y))

print(rf.score(test_x, test_y))
# mean_squared_error(y_true, y_pred, squared=False)
train_y_pred = rf.predict(train_x)
test_y_pred = rf.predict(test_x)
print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y, train_y_pred),
                                   r2_score(test_y, test_y_pred)))

print ('MSE train: %.3f, test: %.3f' % (mean_squared_error(train_y, train_y_pred),
                      mean_squared_error(test_y, test_y_pred)))

print ('MAE train: %.3f, test: %.3f' % (mean_absolute_error(train_y, train_y_pred),
                      mean_absolute_error(test_y, test_y_pred)))

feature_importance=rf.feature_importances_
#make importances relative to max importance
feature_importance=100.0 *(feature_importance.max())
#print(feature_importance)  #use inbuilt class feature_importances

feat_importance=pd.Series(rf.feature_importances_)
feat_importance.nlargest(10).plot(kind='barh')
plt.title('Variable Importance')
plt.xlabel('Relative Importance')
plt.show()