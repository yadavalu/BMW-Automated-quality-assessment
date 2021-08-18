import numpy as np
from PIL import Image


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def sech_squared(x):
    return ((np.exp(x) + np.exp(-x)) ** 2) ** -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dash(x):
    return x * (1 - x)


def error(exp, got, _in, f):
    err = exp - got
    if f == sigmoid: adj = err * sigmoid_dash(got)
    elif f == tanh: adj = err * sech_squared(got)
    return np.dot(_in.T, adj)

def mean_error_function(output, label):
    return 1/len(output) * np.sum((output - label) ** 2, axis=0)

def resize(filename, size=(28, 28)):
    return Image.open(filename).resize(size).save(filename)
