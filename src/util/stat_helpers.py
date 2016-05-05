
import numpy as np


def calculate_correlation_nd(data1, data2, axis=0):
    """
    Calculate correlations of nd arrays along the given axis
    :param data1:
    :param data2:
    :param axis:
    :return:
    """
    nt = data1.shape[axis]
    assert data1.shape == data2.shape

    view1 = data1
    view2 = data2

    if axis:
        view1 = np.rollaxis(data1, axis)
        view2 = np.rollaxis(data2, axis)

    data1_norm = (view1 - data1.mean(axis=axis)) / data1.std(axis=axis)
    data2_norm = (view2 - data2.mean(axis=axis)) / data2.std(axis=axis)

    return np.sum(data1_norm * data2_norm / float(nt), axis=0)




def test():

    arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    print(calculate_correlation_nd(arr, arr.T, axis=1))


if __name__ == '__main__':
    test()