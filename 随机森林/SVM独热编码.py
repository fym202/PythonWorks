import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt    #画图
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

#定义相关变量
random_1 =35# 样本集选取随机种子
rate_1 = 0.25# 样本集验证集
f_name= "新数据  独热编码.csv"
#读入数据
data =pd.read_csv('新数据  独热编码.csv',encoding='gb2312')
data_x=data[['总建筑面积','总增长成本','单位增长成本','单位节能量','单位节材','单位节水','单位节地',' 废气减排',
            '居民健康','提高就业率','居民幸福感指数']]

data_y=data['效益']#独热编码
encoder = OneHotEncoder()
encoder.fit(data.iloc[:,9:13])
print("被转换的列",data.iloc[:,9:13])
data_transformed=encoder.transform(data.iloc[:,9:13]).toarray()
# print("转换后的数据", data_transformed)
categories=encoder.categories_
feature_names=encoder.get_feature_names_out()
print(categories)
print(feature_names)
#去掉编码的那几列
print("需要去掉的列", data_x.columns[7:11])
data2=data_x.drop(data_x.columns[7:11], axis=1)

# x需要使用被转换后的数据
data_transformed_x = pd.concat([data2, pd.DataFrame(data_transformed)], axis=1) # 将date2和独热编码拼接到一起


#分割训练集（后面的会乘分配率）（train_test_split是分配模块函数）
train_x, test_x, train_y, test_y = train_test_split(data_transformed_x, data_y, test_size=rate_1, random_state=random_1)

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
mm_X.fit(data_transformed_x)
train_x = mm_X.transform(train_x)
test_x = mm_X.transform(test_x)# 网格搜索交叉验证（GridSearchCV）：以穷举的方式遍历所有可能的参数组合
param_grid = {'gamma':[0.1,0.2,1,10,20,30,40,50,60,100,150] ,

            'C': [1,10,100,200,350,300,150,250,400]}
#param_grid = {'gamma':[0.3,0.4,0.5,0.8,1,2,10,20,3,5] ,
             #'C': [0.5,0.1,1,2,5,7,9,11,13,15,20,30,40]}
# # cv，交叉验证参数,750
# rbf_svc_cg = GridSearchCV(SVC(kernel='rbf',degree=50,decision_function_shape='ovr',cache_size=1,max_iter=-1),
#                           param_grid, scoring = 'neg_mean_squared_error',
#                        n_jobs =5, cv =2)
# '''
#  # 网格搜索的模型评分会涉及到利用测试集进行评分，所以将数据集分成三部分
# '''
# rbf_svc_cg.fit(train_x,train_y)
# best_g = rbf_svc_cg.best_params_.get('gamma')
# best_c = rbf_svc_cg.best_params_.get('C')
# # 最优参数
# param_grid = {'gamma': [best_g], 'C': [best_c]}
# print(best_g, best_c)

# #SVR，训练模型的包，rbf_svc，定义训练函数
# rbf_svc = SVC(kernel='rbf',C=best_c,gamma=best_g)
# #训练
# rbf_svc.fit(train_x, train_y)
# rbf_svc = SVC(C=0.1).fit(train_x, train_y)
# #测试集
rbf_svc = MLPClassifier( alpha=1e-4, hidden_layer_sizes=(100, 50),
                     random_state=42, tol=1e-4, max_iter=1000)

rbf_svc.fit(train_x, train_y)
test_y_predict = rbf_svc.predict(test_x)
test_y_predict = np.reshape(test_y_predict, (-1, 1))
train_y_predict = rbf_svc.predict(train_x)
train_y_predict = np.reshape(train_y_predict, (-1, 1))
print(test_y_predict)
print(test_y)



# print(rbf_svc_cg.score(test_x, test_y))
#测试集的验证效果
print('决定系数R^2。The value of R-squared of kernal=rbf is',r2_score(test_y,test_y_predict))
print('均方误差MSE。The mean squared error of kernal=rbf is',mean_squared_error(test_y,test_y_predict))
print('平均绝对误差MAE。The mean absolute error of kernal=rbf is',mean_absolute_error(test_y,test_y_predict))

print('R^2 train: %.3f, test: %.3f' % (r2_score(train_y, train_y_predict),
                                   r2_score(test_y, test_y_predict)))

print ('MSE train: %.3f, test: %.3f' % (mean_squared_error(train_y, train_y_predict),
                      mean_squared_error(test_y, test_y_predict)))

print ('MAE train: %.3f, test: %.3f' % (mean_absolute_error(train_y, train_y_predict),
                      mean_absolute_error(test_y, test_y_predict)))

# 真实/模型_1
plt.plot(train_y, color='g', label='true')
plt.plot(test_y_predict, color='b', label='pre')
plt.xlabel("no.")
plt.ylabel("error(m)")
plt.title('model1')
plt.grid()
plt.legend()
plt.show()

# 真实/模型_2
fig = plt.figure(3)
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(test_y, color='g', label='true')
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(test_y_predict, color='b', label='pre')
plt.show()

# 真实/模型_3
p_x = [x for x in range(int(min(test_y)) - 5, int(max(test_y)) + 5)]
p_y = p_x
plt.plot(p_x, p_y, color='black', label='15')
plt.scatter(test_y_predict,test_y, color='b', marker='x',
            label='15')  # https://www.cnblogs.com/shanlizi/p/6850318.html
plt.xlabel('PRE')
plt.ylabel('DTU')
plt.show()
