import matplotlib.pyplot as plt
import h5py
from dnn_app_utils import *
from PIL import Image

def L_layers_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=2500,print_cost=False):
    """
    Input:
    X -- 数据，numpy数组（样本数，像素*像素*3）
    Y --
    """
    costs = []
    # 初始化
    parameters = initialize_parameters_deep(layer_dims)

    for i in range(0,num_iterations):

        # 前向传播
        AL,caches= L_model_forward(X,parameters)

        # 计算损失函数
        cost = compute_cost(AL,Y)

        # 反向传播
        grads = L_model_backward(AL,Y,caches)

        # 更新参数
        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i% 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations(per tens)')
    plt.title("learning_rate="+str(learning_rate))
    plt.show()

    return parameters