
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
#使用机器学习库中的线性模型进行对比
from sklearn import linear_model

#sigmod函数
def sigmod(x):
    res=1/(1+np.exp(-x))
    return res




# def J_cost(X, y, beta):
#     '''
#     :param X:  sample array, shape(n_samples, n_features)
#     :param y: array-like, shape (n_samples,)
#     :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
#     :return: the result of formula 3.27
#     '''
#     X_hat = np.c_[X, np.ones((X.shape[0], 1))]
#     beta = beta.reshape(-1, 1)
#     y = y.reshape(-1, 1)

#     Lbeta = -y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta)))

#     return Lbeta.sum()

#预测函数，进行模型二分类
def predict(X,beta):
    X_new=np.c_[X,np.ones(X.shape[0],1)]
    p1=sigmod(np.dot(X_new,beta))
    if p1>=0.5:
        return True
    else:
        return False








#对数几率回归(使用梯度下降法迭代)
def Logistic_Regression(X,y,iteration_times,learning_rate):

    #首先，我们需要随机初始化β，由于x为2维，则β为2+1维，全部初始化为1
    # beta=np.ones((X.shape[1]+1,1))
    beta = np.random.randn(X.shape[1] + 1, 1) * 0.5 + 1
    
    for i in range(iteration_times):
        #首先将X变为增广矩阵，即添加一个常数列
        X_new=np.c_[X,np.ones((X.shape[0],1))]
        beta = beta.reshape(-1, 1)
        y = y.reshape(-1, 1)
        #向量内积

        p1=sigmod(np.dot(X_new,beta))
        #计算梯度
        grad=np.dot(-X_new.T,(y-p1))


        #梯度下降
        beta=beta-(learning_rate*grad)
       
    return beta




if __name__=='__main__':
    data=pd.read_csv('./DATA.csv').values
    #检查是否读取正确
    print(data.shape)
    #获取数据
    #X矩阵 分别有密度和含糖率
    X=data[:,0:2].astype(float)
    y=data[:,2].astype(int)
    print(X)
    print(y)
    beta=Logistic_Regression(X,y,1000,0.3)
    print(beta)
    x0=np.linspace(0,1,100)
    y0=-(beta[0] * x0 + beta[2]) / beta[1]

    plt.plot(x0, y0, label=r'Logistic_Regression_By')
    plt.show();






