import numpy as np
from PIL import Image

e = 2.718281828

def sigmoid(x):
    return 1 / (1 + e ** (-x))

def sigmoid_dash(x):
    return x * (1 - x)

def tanh(x):
    return (e ** x - e ** (-x))/(e ** x + e ** (-x))

def sech_squared(x):
    return ((e ** x + e ** (-x)) ** 2) ** -1

def error(exp, got, _in):
    err = exp - got
    adj = err * sigmoid_dash(got)

    return np.dot(_in.T, adj)


def mean_error_function(output, label):
    return 1/len(output) * np.sum((output - label) ** 2, axis=0)


def resize(filename, size=(28, 28)):
    return Image.open(filename).resize(size).save(filename)
