import numpy as np
from sklearn.datasets import load_boston

data = load_boston()
x_ = data['data']
y = data['target']

y = y.reshape(y.shape[0],1)

x_ = (x_ - np.mean(x_,axis = 0))/np.std(x_,axis = 0)

n_facture = x_.shape[1]
h_hidden = 10
w1 = np.random.randn(n_facture,h_hidden)
b1 = np.zeros(h_hidden)
w2 = np.random.randn(h_hidden,1)
b2 = np.zeros(1)

def MSE_loss(y,y_hat):
    return np.mean(np.square(y_hat-y))

def linear(x,w,b):
    y = x.dot(w)+b
    return y

def Relu(x_):
    r = np.where(x_<0,0,x_)
    return r

#前向传播
learning_rate = 1e-6
for i in range(5000):
    l1 = linear(x_,w1,b1)
    s1 = Relu(l1)
    y_pred = linear(s1,w2,b2)
    loss = MSE_loss(y,y_pred)

    #反向传播
    grad_y_pred = 2*(y_pred - y)
    grad_w2 = s1.T.dot(grad_y_pred)
    grad_temp_relu = grad_y_pred.dot(w2.T)
    grad_temp_relu[l1<0]=0
    grad_temp = grad_temp_relu.copy()
    grad_w1 = x_.T.dot(grad_temp)

    w1 -= grad_w1*learning_rate
    w2 -= grad_w2*learning_rate

print(y)
print(y_pred)
print(y_pred-y)




