import os

import numpy as np
from matplotlib import pyplot as plt

folder = 'saved/HopperBulletEnv-v0-homerun/fits'
y = []
x = []
for i, file in enumerate(os.listdir(folder)):
    loaded = np.load(f'{folder}/{file}')
    y += list(loaded)
    x += [i] * len(loaded)

plt.scatter(x, y)
