__author__ = 'huziy'

import numpy as np

def prev_func(r):
    return 2.576e-6 * r ** 2 + 0.074

def sturm_func(r, tempK = 273):
    if r > 156:
        x = 0.138-1.010 * (r / 1000.0) + 3.233 * (r/1000.0) ** 2
    else:
        x = 0.023 + 0.234 * (r / 1000.0)

    return x + 2.7e-4 * 2 ** ((tempK - 233) / 5.0)




def main():
    import matplotlib.pyplot as plt
    fig = plt.figure()

    rs = np.arange(1, 500, 1)

    old_snc = list(map(prev_func, rs))
    sturm_snc = list(map(sturm_func, rs))
    plt.plot(rs, old_snc, label = "old")
    plt.plot(rs, sturm_snc, label = "new")
    plt.legend()

    plt.show()

    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  