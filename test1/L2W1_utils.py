import numpy as np

def initialize_parameters_zeros(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range (1, L):
        parameters["W"+str(l)] = np.zeros((layer_dims[l],layer_dims[l-1]))
        parameters["b"+str(l)] = np.zeros((layer_dims[l],1))

        assert(parameters["W"+str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert(parameters["b"+str(l)].shape == (layer_dims[l],1))

    return parameters

def initialize_parameters_random(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L + 1):
        parameters["W" + str(l)] = np.random.randn((layer_dims[l], layer_dims[l - 1]))*10
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layer_dims[l], 1))

    return parameters

def initialize_parameters_he(layer_dims):

    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn((layer_dims[l], layer_dims[l - 1]))*np.sqrt(2./layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters["b" + str(l)].shape == (layer_dims[l], 1))

    return parameters

parameters = initialize_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))