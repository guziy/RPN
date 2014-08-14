__author__ = 'huziy'


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

def vel(z):
    return np.exp(-z)

def animate(i):

    print animate.x

    pass


def main():
    fig = plt.figure()
    animate.x = [0] * 20
    animate.y = np.linspace(0, 1, 20)
    animate.dt = 0.1

    ani = animation.FuncAnimation(fig, animate)
    plt.show()





if __name__ == "__main__":
    pass