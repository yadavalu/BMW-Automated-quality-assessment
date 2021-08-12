import os

#from PIL import Image
from matplotlib.image import imread
import numpy as np

import activation as a


def load_data(index):
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

    #print(images[0])
    #print(crack)

    return images, crack


def cache(file: str): 
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
    f = open(file)
    for i in ls: 
        for j in i:
            f.write(str(j))
    f.close()

def main():
    for i in range(1, 5):
        print(f"Loading data set {i} ...")
        data, crack = load_data(i)

        rate = 0.01
        epochs = 5

        if os.path.exists("cache/weights.cache"):
            print("Caching weights ...")
            weights = cache("cache/weights.cache")
        else:
            print("Generating random weight ...")
            np.random.seed(0)
            weights = [
                np.random.uniform(-1, 1, (50, 227**2)),
                np.random.uniform(-1, 1, (1, 50)),
            ]

        if os.path.exists("cache/bias.cache"):
            print("Caching bias ...")
            bias = cache("cache/bias.cache")
        else:
            print("Generating random bias ...")
            np.random.seed(0)
            bias = [
                np.zeros((50, 1)),
                np.zeros((1, 1)),
            ]

        for epoch in range(1, epochs + 1):
            print(f"{epoch = }")
            correct = 0

            for d, c in zip(data, crack): 
                # Forward propagation
                d.shape += (1,)
                c.shape += (1,)

                h_neuron = a.tanh(weights[0] @ d + bias[0])
                o_neuron = a.tanh(weights[1] @ h_neuron + bias[1])

                error = a.mean_error_function(o_neuron, c)
                correct += int(c == np.argmax(o_neuron))

                do = o_neuron - c
                dh = np.transpose(weights[1]) @ do * a.sech_squared(h_neuron)

                weights[1] += -rate * do @ np.transpose(h_neuron)
                bias[1] += -rate * do

                weights[0] += -rate * dh @ np.transpose(d)
                bias[0] += -rate * dh

            print(f"Neural network accuracy = {round(correct * 100/40000, 2)}")
            print(f"Correct = {correct}/40000")

if __name__ == "__main__":
    main()
