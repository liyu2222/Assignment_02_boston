import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.array([50,80,100,200])
y = np.array([82,118,172,302])
plt.scatter(x,y) #生成scatter散点图
# plt.grid()  #生成网格
# plt.show()

#规范化,数据规范化
max_x = np.max(x)
max_y = 1
x = x/max_x
y = y/max_y

def MAE_loss(y,y_hat):
    return np.mean(np.abs(y_hat - y))

def MSE_loss(y,y_hat):
    return np.mean(np.square(y_hat-y))

# y_hat = np.array([-2,-1,-1,3,5])
# print(MAE_loss(y,y_hat))

def linear(x,k,b):
    y = k*x + b
    return y

#暴力穷举
# min_loss = float('inf') #inf 代表正无穷
# for k in np.arange(-2,2,0.1):
#     for b in np.arange(-10,10,0.1):
#         y_hat = [linear(xi,k,b) for xi in list(x)]
#         current_loss = MSE_loss(y,y_hat)
#         if current_loss < min_loss:
#             min_loss = current_loss
#             best_k,best_b = k,b
#             print('best_k is {},best_b is {}'.format(best_k,best_b))


#梯度下降  对mse_loss求梯度，将所有的样本求和再取平均
def gradiant_k(x,y,y_hat):
    gradiant = 0
    for xi,yi,yi_hat in zip(list(x),list(y),list(y_hat)):
        gradiant += (yi_hat - yi)*xi
    return gradiant/len(x)
def gradiant_b(x,y,y_hat):
    gradiant = 0
    for xi,yi,yi_hat in zip(list(x),list(y),list(y_hat)):
        gradiant += (yi_hat-yi)
    return gradiant/len(x)


min_loss = float('inf')
try_time = 1000
learn_rate = 0.1
current_k = 10
current_b = 10
for i in range(try_time):
    y_hat = [linear(xi,current_k,current_b) for xi in list(x)]
    current_loss = MSE_loss(y,y_hat)
    if current_loss < min_loss:
        min_loss = current_loss
        best_k,best_b = current_k,current_b
    current_k -= gradiant_k(x,y,y_hat)*learn_rate   #learn_rate学习率
    current_b -= gradiant_b(x,y,y_hat)*learn_rate

best_k = best_k/(max_x*max_y)
best_b = best_b/(max_x*max_y)
print(best_k,best_b)

x = x*max_x
y = y*max_y
y_hat = best_k *x +best_b
plt.plot(x,y_hat,color= 'red')
plt.grid()
plt.show()