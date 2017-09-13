

import numpy as np
import pandas as pd


def agg_blocks(data, block_shape: tuple, func=np.mean):

    assert data.ndim == len(block_shape)

    indices = np.indices(data.shape)

    ser = pd.Series(data=data.flatten(), index=zip(indices[0].flatten(), indices[1].flatten()))

    agg = ser.groupby(lambda ind: tuple(ind[j] // block_shape[j] for j in range(len(block_shape)))).apply(func)

    new_data = np.ma.masked_all(tuple(data.shape[j] // block_shape[j] + int(data.shape[j] % block_shape[j] != 0) for j in range(len(block_shape))))

    i_arr, j_arr = zip(*agg.index)
    i_arr = np.array(i_arr)
    j_arr = np.array(j_arr)

    new_data[i_arr, j_arr] = agg[agg.index].values
    return new_data



def test_agg_blocks():
    agg_blocks(np.random.randn(20, 20), (2,  2))
    agg_blocks(np.random.randn(20, 20), (5,  5))
    agg_blocks(np.random.randn(20, 20), (7,  7))


    x = np.random.randn(5, 5)
    xm = np.ma.masked_where(x < 0, x)

    def __agg(arr):
        assert isinstance(arr, pd.Series)
        v = arr.values

        print(v.shape, v, v.mask)

        bad = v.mask | np.isnan(v)

        if not np.all(bad):
            return v[~bad].mean()
        else:
            return np.ma.masked

    agg1 = agg_blocks(xm, (2, 2), func=__agg)
    print(xm)
    print(agg1)


if __name__ == '__main__':
    pass