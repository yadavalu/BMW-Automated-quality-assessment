import os

from matplotlib.image import imread
import numpy as np

import activation as a


def load_data(index):
    """
    returns data
    :param:  index: data set number
    :return: data
    """

    images, crack = [], []

    for file in os.listdir(f"data/{index}/+/"):
        images.append(
            imread(f"data/{index}/+/" + file).astype("float32") / 255
        )
        crack.append(1)

    for file in os.listdir(f"data/{index}/-/"):
        images.append(
            imread(f"data/{index}/-/" + file).astype("float32") / 255
        )
        crack.append(-1)

    return images, crack


def cache(file: str): 
    """
    reads cached file
    :param:  file
    :return: layer
    """

    ret = [[]]
    layer = []

    f = open(file, "r")
    for line in f.readlines():
        if line == ",":
            ret.append(layer)
            layer = []
        else:
            layer.append(float(line))
    f.close()

    return ret

def write_cache(ls, file: str):
    """
    writes cache into cache file
    :param: layer, file
    """
    
    f = open(file)
    for i in ls: 
        for j in i:
            f.write(str(j))
    f.close()

def main():
    """
    main function
    """
    for i in range(1, 5):
        print(f"Loading data set {i} ...")
        data, crack = load_data(i) # load data into memory

        rate = 0.01 # learning rate
        epochs = 5 # number of guesses

        # open cached weights
        if os.path.exists("cache/weights.cache"):
            print("Caching weights ...")
            weights = cache("cache/weights.cache")
        else: # else regenerate
            print("Generating random weight ...")
            np.random.seed(0)
            weights = [
                np.random.uniform(-1, 1, (50, 227**2)),
                np.random.uniform(-1, 1, (1, 50)),
            ]

        # open cached biases
        if os.path.exists("cache/bias.cache"):
            print("Caching bias ...")
            bias = cache("cache/bias.cache")
        else: # else regenerate
            print("Generating random bias ...")
            np.random.seed(0)
            bias = [
                np.zeros((50, 1)),
                np.zeros((1, 1)),
            ]

        for epoch in range(1, epochs + 1):
            # train
            print(f"{epoch = }")
            correct = 0

            for d, c in zip(data, crack): 
                # forward propagation
                d.shape += (1,)
                c.shape += (1,)

                # run activation function
                h_neuron = a.tanh(weights[0] @ d + bias[0])
                o_neuron = a.tanh(weights[1] @ h_neuron + bias[1])

                # calculate error
                error = a.mean_error_function(o_neuron, c, a.tanh)
                correct += int(c == np.argmax(o_neuron))

                # back propogation
                do = o_neuron - c
                dh = np.transpose(weights[1]) @ do * a.sech_squared(h_neuron)

                # calculate weights and biases
                weights[1] += -rate * do @ np.transpose(h_neuron)
                bias[1] += -rate * do

                weights[0] += -rate * dh @ np.transpose(d)
                bias[0] += -rate * dh

            # verbose
            print(f"Neural network accuracy = {round(correct * 100/40000, 2)}")
            print(f"Correct = {correct}/40000")

if __name__ == "__main__":
    main()
