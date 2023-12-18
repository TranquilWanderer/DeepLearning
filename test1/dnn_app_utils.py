import numpy as np
import h5py

def sigmoid(z):
    A = 1/(1 + np.exp(-z))
    cache = z
    return A,cache

def relu(z):

    A = np.maximum(0,z)
    assert (A.shape == z.shape)

    cache = z
    return A,cache

def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA,copy=True)
    dZ[Z <= 0] = 0

    assert(dZ.shape == Z.shape)

    return dZ

def sigmoid_backward(dA,cache):
    '''激活函数部分的反向传播，获取dZ,
        此处的cache = activation_cache,即Z
    '''
    Z = cache

    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1 - s)

    assert (dZ.shape == Z.shape)

    return dZ


def initialize_parameters(n_x, n_h, n_y):
    ''' 初始化两层神经网络 '''
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))

    assert (W1.shape == (n_h,n_x))
    assert (W2.shape == (n_y,n_h))
    assert (b1.shape == (n_h,1))
    assert (b2.shape == (n_y,1))

    parameters = {"W1":W1,
                 "b1":b1,
                 "W2":W2,
                 "b2":b2}
    return parameters

def initialize_parameters_deep(layer_dims):
    '''将神经网络初始化扩展到L层'''
    # 变化出现在需要将输入扩展成列表 （n_x,n_h,n_y）——>layer_dims

    L = len(layer_dims)  #神经网络的层数
    parameters = {}
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1]) #*0.01
        parameters["b"+str(l)] = np.zeros((layer_dims[l],1))

        assert (parameters["W"+str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert (parameters["b"+str(l)].shape == (layer_dims[l],1))

    return parameters

def linear_forward(A,W,b):
    '''线性输入正向传播'''

    Z = np.dot(W,A)+b
    assert (Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)

    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    '''正向线性——>正向激活，将两个合在一起'''
    # linear——>activation 一层
    '''输入：输入数据/前一层的激活值
            W
            b
            激活函数
    '''
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A,cache

def L_model_forward(X,parameters):
    '''执行整个神经网络的正向传播'''
    # [linear-->activation]------>[linear-->activation]->linear->sigmoid
    # 一个方括号是一层，每层分两步
    # 输入X=A0，W,b即parameters
    caches = []
    A = X
    L = len(parameters)//2
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation='relu')
        caches.append(cache)
    # 在for循环结束时候，此时的A实际已经更新到A(L-1)了，所以下面接着A
    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation='sigmoid')
    caches.append(cache)

    assert (AL.shape == (1,X.shape[1]))

    return AL,caches
'''
在给出的 L_model_forward 函数中，caches 是一个列表（list），其中包含了每一层的前向传播过程中产生的缓存信息。
缓存信息用于存储当前层的线性部分的输入数据、权重和偏置，以及激活函数的输出，这些信息在反向传播时是必需的。

caches 列表中的每个元素都是一个元组（tuple），其中包含了两个元素：linear_cache 和 activation_cache。其中：
linear_cache 包含了线性步骤的输入: A^[l-1],  W^[l] , b^[l]。
activation_cache 包含了线性步骤的输出 Z^[l]，即激活函数的输入。
对于 [LINEAR -> RELU] 的 L-1 层，每个缓存都会添加到 caches 列表中，最后一层的 [LINEAR -> SIGMOID] 产生的缓存也会被添加到列表中。

下面是一个简化的例子来帮助理解 caches 的结构：
假设我们有一个3层的神经网络，那么 caches 列表将包含3个元素，每个元素对应一层的缓存信息。如果我们展开它，可能看起来像这样：
caches = [
    # Cache from layer 1 (linear_relu_forward)
    (
        (X, W1, b1),  # linear_cache
        Z1             # activation_cache
    ),
    # Cache from layer 2 (linear_relu_forward)
    (
        (A1, W2, b2),  # linear_cache
        Z2             # activation_cache
    ),
    # Cache from layer 3 (linear_sigmoid_forward)
    (
        (A2, W3, b3),  # linear_cache
        Z3             # activation_cache
    )
]
'''

def compute_cost(AL,Y):
    m = Y.shape[1]
    #cost = (-1./m)*(np.dot(Y, np.log(AL).T)+np.dot(1 - Y, np.log(1 - AL).T))
    cost = -1 / m * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL),axis=1,keepdims=True)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    '''
    注：a = np.array([[1,2,3,4]]) b = np.array([1,2,3,4]) c = np.array([[1,1,1,1]])
       d = np.array([1,1,1,1])   e = np.dot(a,c.T)       f = np.dot(b,d)
       print(e)       print(f)
       [[10]]            10
    
    '''
    return cost

def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = 1/m*np.dot(dZ,A_prev.T)
    db = 1/m*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)

    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    '''反向传播一层'''

    linear_cache,activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)

    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL,Y,caches):
    '''L层的反向传播，需要用到L层正向传播的caches
       输入：AL，Y，caches
       输出：grads = {}梯度
    '''
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]  # 第L个缓存值，其在列表的索引为L-1
    grads['dA'+str(L-1)] ,grads['dW'+str(L)],grads['db'+str(L)] = linear_activation_backward(dAL,current_cache,activation='sigmoid')

    for l in reversed(range(L-1)):
        # 第l层：relu -> linear
        # 索引：(0,1,2,3,...,L-2,)=> (L-2,L-3,...,2,1,0)
        current_cache = caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads['dA'+str(l+1)],current_cache,activation='relu')
        grads['dA'+str(l)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp      # 即dW^[L-1]
        grads['db'+str(l+1)] = db_temp

    return grads

def update_parameters(parameters,grads,learning_rate):
    '''
    更新参数
    输入：parameters,grads,learning_rate
    输出：parameters
    '''

    L = len(parameters)//2

    for l in range(1,L+1):
        parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate*grads["dW"+str(l)]
        parameters['b'+str(l)] = parameters['b'+str(l)] - learning_rate*grads['db'+str(l)]

    return parameters

'''def predict(X,Y,parameters):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    Z1 = np.dot(parameters["W1"],X) + parameters["b1"]
    A1,cache1 = relu(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2,cache2 = sigmoid(Z2)

    for i in range(A2.shape[1]):
        if A2[0,i] < 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    assert (Y_prediction.shape == (1,m))

    correct_predictions = np.sum(Y_prediction == Y)
    accuracy = correct_predictions / m

    return Y_prediction,accuracy
'''

def predict(X,Y,parameters):


    m = X.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    # forward_propagation
    probas, caches = L_model_forward(X, parameters)

    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    print("准确度为：" + str(float(np.sum((p == Y)) / m)))

    return p


def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes