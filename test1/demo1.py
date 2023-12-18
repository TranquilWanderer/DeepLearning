from dnn_app_utils import *
def compute_cost_with_regularization(AL,Y,parameters,lambd):
    '''
    L2 regularization
    :param AL:
    :param Y:
    :param parameters:
    :param lambd:
    :return:cost
    '''
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(AL,Y)
    L2_regularization_cost = lambd/(2*m)*(np.sum(np.square(W1))+np.sum(np.squre(W2))+np.sum(np.squre(W3)))

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_propagation_with_regularization(X,Y,cache,lambd):
    '''

    :param X:
    :param Y:
    :param cache:
    :param lambd:
    :return:
    '''
    m = Y.shape[1]
