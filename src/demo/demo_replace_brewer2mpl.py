from matplotlib.colors import LinearSegmentedColormap

__author__ = 'huziy'
import matplotlib as mpl
import matplotlib.pyplot as plt
import brewer2mpl
import numpy as np
from matplotlib import cm

# Conclusion: Could not find an easy way to generate the colorcycle using only mpl
# so sticking to brewer2mpl


def generate_series():
    return [np.random.randn(20) for _ in range(5)]


def plot_data(data):
    plt.figure()
    for i, ser in enumerate(data):
        plt.plot(ser, label="{}".format(i))


def main():
    data = generate_series()

    print(mpl.rcParams["axes.color_cycle"])

    # brewer version

    # brewer2mpl.get_map args: set name  set type  number of colors
    bmap = brewer2mpl.get_map("Set1", "qualitative", 9)
    # Change the default colors
    mpl.rcParams["axes.color_cycle"] = bmap.mpl_colors
    plot_data(data)
    print(bmap.mpl_colors)


    # mpl version

    cmap = cm.get_cmap("Set1")
    assert isinstance(cmap, LinearSegmentedColormap)
    print(type(cmap))

    #mpl.rcParams["axes.color_cycle"] =
    plot_data(data)

    plt.show()

    pass


if __name__ == '__main__':
    main()
