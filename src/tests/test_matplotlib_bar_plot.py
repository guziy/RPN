
import matplotlib.pyplot as plt

import matplotlib

from matplotlib.patches import Patch

def main():

    print(matplotlib.__version__)
    fig = plt.figure()
    ax = plt.gca()

    data = list(range(5, 10))

    print(matplotlib.get_backend())

    ax.bar(data, data, width=0.5, align="edge", linewidth=[2] * len(data),
           edgecolor=["k"] * len(data), facecolor="m")

    plt.show()


if __name__ == '__main__':
    main()