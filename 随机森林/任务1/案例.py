import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt    #画图
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, roc_auc_score, auc
from  sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import utils


# 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#定义相关变量
random_1 =5# 样本集选取随机种子
#rate_1 = 0.25# 样本集验证集
f_name= "案例数据.csv"
#读入数据
data =pd.read_csv('案例数据.csv',encoding='gb2312')
data_z=data[['总建筑面积','单位增长成本','单位节能','单位节材','单位节水','单位节地','CO2减排效益','健康效益','提高就业率效益',
             '节省排污、设施费']]

#独热编码
encoder = OneHotEncoder()
encoder.fit(data.iloc[:,6:10])
print("被转换的列",data.iloc[:,6:10])
data_transformed=encoder.transform(data.iloc[:,6:10]).toarray()
print("转换后的数据", data_transformed)
categories=encoder.categories_
feature_names=encoder.get_feature_names_out()
print(categories)
print(feature_names)
#去掉编码的那几列
print("需要去掉的列", data_z.columns[6:10])
data2=data_z.drop(data_z.columns[6:10], axis=1)

# x需要使用被转换后的数据
data_transformed_z = pd.concat([data2, pd.DataFrame(data_transformed)], axis=1) # 将date2和独热编码拼接到一起
#分割训练集（后面的会乘分配率）（train_test_split是分配模块函数）
#train_x, test_x, train_y, test_y = train_test_split(data_transformed_x, data_y, test_size=rate_1, random_state=random_1)

#将用到的数据集转变成数组的格式
train_z=np.array(train_z)

test_z=np.array(test_z)

print(train_z.shape)

print(test_z.shape)


#对标签的处理
train_y = np.reshape(train_y, -1)
test_y = np.reshape(test_y, (-1,1))
print(train_y.shape)
print(test_y.shape)
#进行数据的归一化,得到归一标准化后的值
mm_X = StandardScaler()
mm_X.fit(data_transformed_z)
train_z = mm_X.transform(train_z)
test_z = mm_X.transform(test_z)

##随机森林分类器

rf=RandomForestRegressor(n_estimators=58,max_depth=None,min_samples_leaf=1,min_samples_split=2,
                         max_features=6,random_state=37,min_weight_fraction_leaf=0.01)


print(rf.fit(train_z, train_y))

print(rf.score(test_z, test_y))
#mean_squared_error(y_true, y_pred, squared=False)
train_y_pred = rf.predict(train_z)
test_y_pred = rf.predict(test_z)
print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y, train_y_pred),
                                   r2_score(test_y, test_y_pred)))

print ('MSE train: %.3f, test: %.3f' % (mean_squared_error(train_y, train_y_pred),
                      mean_squared_error(test_y, test_y_pred)))

print ('MAE train: %.3f, test: %.3f' % (mean_absolute_error(train_y, train_y_pred),
                      mean_absolute_error(test_y, test_y_pred)))