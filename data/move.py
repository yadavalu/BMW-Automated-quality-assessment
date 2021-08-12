#!/usr/bin/env python3

import os

dir = ["Positive", "+"]

for i in range(2):
    for j in range(1, 5):
        for k in os.listdir(str(j)):
            for l in range(4000):
                os.system(f"cp {dir[0]}/{str(j * 4000 + l).zfill(5)}{'_1' if dir[1] == '+' and 19378 >= (j * 4000 + l) >= 10000 else ''}.jpg {j}/{dir[1]}/")
                #print(str((j - 1) * 4000 + l).zfill(5))

    dir = ["Negative", "-"]
