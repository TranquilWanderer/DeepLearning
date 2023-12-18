import numpy as np
import h5py

    
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(z):
    A = 1/(1 + np.exp(-z))
    cache = z
    return A,cache

def relu(z):

    A = np.maximum(0,z)
    assert (A.shape == z.shape)

    cache = z
    return A,cache
def linear_forward(A,W,b):

    Z = np.dot(W,A)+b
    assert (Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)

    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A,cache

def forward_propagation(X,parameters):

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

