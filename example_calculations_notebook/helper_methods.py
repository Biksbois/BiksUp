import numpy as np

def sigmoid(num):
    return 1 / (1+np.exp(-num))

def activation_function_on_vector(func, input_vector):
    return [func(i) for i in input_vector]

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t #,dt

def get_initial_weight(matrix):
    return np.random.rand(matrix.shape[0], matrix.shape[1]) 