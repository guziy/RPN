__author__ = 'huziy'


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    path = "/skynet2_exec2/sca7026-30/huziy/QUEBEC.OF4"

    data = pd.read_fwf(path, skiprows=3, usecols=(4, 5, 6, 7, 9), header=None,
                       widths=[3] * 2 + [5, 6] + 9 * [8, ] + 2 * [7, ] + [11, 8, 12])

    print(data.head())

    labels = ["SWnet", "LWnet", "SH", "LH", "G"]


#    for label, flx in zip(labels, data.values.T):
#        plt.plot(flx, label=label, lw=2)

    print(data.values.shape)

    bal = data.values[:, 0] + data.values[:, 1] - data.values[:, 2] - data.values[:, 3] - data.values[:, 4]

    plt.plot(bal, label="balance", linewidth=3)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
