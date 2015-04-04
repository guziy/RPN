__author__ = 'huziy'


import numpy as np
from numpy import random


def main():
    from scipy import sparse
    data = random.randn(5, 5)
    good = data > 0.5
    i, j = np.where(good)


    mat1 = sparse.coo_matrix((data[good], (i, j)), shape=data.shape)
    mat2 = sparse.coo_matrix((data[good], (i, j)), shape=data.shape)
    print(mat1.toarray())
    print(mat1.tocsr()[0, 0])

    print(10 * "*")

    print(mat1 + mat2 * 5)

    print(np.prod((5, 6, 7)))


if __name__ == "__main__":
    main()