__author__ = 'huziy'

import numpy as np

x = np.linspace(0, 10, 20)
y = np.linspace(5, 15, 20)

z = np.maximum(x, 100)

import matplotlib.pyplot as plt
plt.plot(x, label="x")
plt.plot(y, label="y")
plt.plot(z, label="z")
plt.legend()
plt.show()
