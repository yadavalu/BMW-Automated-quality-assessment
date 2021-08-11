import os

from PIL import Image
import numpy as np


def load_data():
    images, crack = [], []

    for folders in os.listdir("data/"):
        for files in os.listdir("data/" + folders):
            images.append(files)
            crack.append(folders)

    #print(images)
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
    data, crack = load_data()
    
    rate = 0.01
    epochs = 5

    if os.path.exists("cache/weights.cache"):
        weights = cache("cache/weights.cache")
    else:
        weights = [
            np.random.uniform(-0.5, 0.5, (50, 227**2)),
            np.random.uniform(-0.5, 0.5, (1, 50))
        ]

    if os.path.exists("cache/bias.cache"):
        bias = cache("cache/bias.cache")
    else:
        bias = [
            np.random.uniform(-0.5, 0.5, (50, 227**2)),
            np.random.uniform(-0.5, 0.5, (1, 50))
        ]

    for epoch in range(epochs):
        # TODO
        pass

if __name__ == "__main__":
    load_data()
