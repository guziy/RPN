from matplotlib.ticker import MaxNLocator

__author__ = 'huziy'

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 20, 1)
y = np.arange(0, 10, 0.5)

xx, yy = np.meshgrid(x, y)

z = xx ** 2 + yy ** 2

im = plt.contourf(xx, yy, z)

ax = plt.gca()

ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

plt.savefig("test_contourf.pdf")
