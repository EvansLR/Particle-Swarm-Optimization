import numpy as np
from sklearn import linear_model
import pandas as pd


#将3_3中我们构造的对率回归复制过来

#sigmod函数
def sigmod(x):
    res=1/(1+np.exp(-x))
    return res


#预测函数，进行模型二分类
def predict(X,beta):
    X_new=np.c_[X,np.ones((X.shape[0],1))]
    #在此处使用sigmod发现容易溢出，所以更改判断方式
    p1=sigmod(np.dot(X_new,beta))
    p1[p1>=0.5] = 1
    p1[p1<0.5] = 0
    return p1

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



#精确度统计：根据预测结果和真实结果
def Accuracy(y,y_pred):
    count=0
    for i in range(y.shape[0]):
        if y_pred[i]==y[i]:
            count=count+1
    return count/y.shape[0]







#10折交叉验证法
def my_KFold(X,y,k=10):
    #10次验证准确率累加
    sum_accu=0
    split_count=int(X.shape[0]/k)
    
    for i in range(k):
        test_index=range(i*split_count,(i+1)*split_count)
        #分离出测试集
        X_test=X[test_index]
        y_test=y[test_index]
        #分离出训练集
        X_train=np.delete(X,test_index,axis=0)
        y_train=np.delete(y,test_index,axis=0)
        beta=Logistic_Regression(X_train,y_train,1000,0.25)
        y_pred=predict(X_test,beta)
        
        accu=Accuracy(y_test,y_pred)
 
        sum_accu=sum_accu+accu
    
    return sum_accu/10

#LOO留一法（只留下一个作为测试用例）








if __name__=='__main__':
    #数据处理
    a_data=pd.read_csv('./iris.data').values
    flowers=['Iris-setosa','Iris-versicolor','Iris-virginica']
    #将每一类分别去除，因此可以分为三个二分类问题
    e_accu=0
    for e in range(0,3):
        data=a_data[a_data[:,4]!=flowers[(e+2)%3]]
        #分离出X和Y
        X=data[:,0:3].astype('float')
        X = (X - X.mean(0)) / X.std(0)
        y=data[:,4]
        #按照两种类型进行二分类
        y[y == flowers[(e)%3]] = 1
        y[y == flowers[(e+1)%3]] = 0
        y = y.astype(int)
        #此时要注意，由于数据太过整齐，所以我们需要洗牌
        # shuffle
        index = np.arange(X.shape[0])
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
        n_accu=my_KFold(X,y)
        e_accu=e_accu+n_accu

        print('数据集为',flowers[(e)%3],flowers[(e+1)%3],'时准确率为',n_accu)
    e_accu=e_accu/3
    print('总准确率为',e_accu)









