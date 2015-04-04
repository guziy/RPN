from datetime import datetime, timedelta
from matplotlib.axes import Axes
from matplotlib.dates import date2num, DateFormatter, MonthLocator

__author__ = 'huziy'

import matplotlib.pyplot as plt
import numpy as np


def demo():
    fig = plt.figure()
    d0 = datetime(2001, 1, 1)
    dt = timedelta(days=1)
    dates = [
        d0 + i * dt for i in range(365)
    ]
    nz = 20
    data = np.random.randn(365, nz)

    z = list(range(nz))
    numdates = date2num(dates)

    z, t = np.meshgrid(z, numdates)

    ax = plt.gca()
    assert isinstance(ax, Axes)
    img = ax.pcolormesh(t, z, data)
    ax.set_xlim([numdates[0], numdates[-1]])
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    ax.xaxis.set_major_locator(MonthLocator())

    plt.colorbar(img)
    plt.show()



if __name__ == "__main__":
    demo()
    pass