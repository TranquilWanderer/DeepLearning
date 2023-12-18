import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils import *
from lr_utils import *

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

# 查看数据集的一些数据

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# 重构训练样本和测试样本
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
# 标准化数值特征，使其在0-1之间
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

n_x = 12288
n_h = 7
n_y =1
layer_dims = [12288,7,1]

def two_layers_model(X,Y,layer_dims,learning_rate=0.005,num_iterations=3000,print_cost=False):

    grads = {}
    costs = []
    # '初始化参数'
    parameters = initialize_parameters(layer_dims[0],layer_dims[1],layer_dims[2])

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(0,num_iterations):
        A1,cache1 = linear_activation_forward(X,W1,b1,activation='relu')
        A2,cache2 = linear_activation_forward(A1,W2,b2,activation='sigmoid')

        cost = compute_cost(A2,Y)
        costs.append(cost)
        dA2 = -(np.divide(Y,A2)-np.divide(1-Y,1-A2))

        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,activation='sigmoid')
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,activation='relu')

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)


    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

parameters = two_layers_model(train_x, train_y, layer_dims, num_iterations = 5000, print_cost=True)

Y_prediction = predict(test_x,test_y,parameters)


