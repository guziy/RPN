

import numpy as np
import pandas as pd
from numba import jit


def agg_blocks(data, block_shape: tuple, func=np.mean):

    assert data.ndim == len(block_shape)

    indices = np.indices(data.shape)

    new_data = np.ma.masked_all(tuple(data.shape[j] // block_shape[j] + int(data.shape[j] % block_shape[j] != 0) for j in range(len(block_shape))))

    # check where the good points are
    good = ~np.isnan(data)
    if hasattr(data, "mask"):
        good = (~data.mask) & good

    if not np.any(good):
        return new_data

    data_1d = data[good]
    if data.ndim > 1:
        inds_1d = zip(*[ax_inds[good] for ax_inds in indices])
    else:
        inds_1d = [(i, ) for i in indices[0]]

    ser = pd.Series(data=data_1d, index=inds_1d)

    agg = ser.groupby(lambda ind: tuple(ind[j] // block_shape[j] for j in range(len(block_shape)))).apply(func)

    if data.ndim > 1:
        indices_out = zip(*agg.index)
        indices_out = tuple(np.array(i) for i in indices_out)
    else:
        indices_out = agg.index

    new_data[indices_out] = agg[agg.index].values

    return new_data



def test_agg_blocks():

    x = np.round(np.random.randn(20, 20), decimals=2)
    print(x)

    aggr = np.round(agg_blocks(x, (2,  2)), decimals=2)
    print(aggr)


    print(aggr[0, 0], x[:2, :2].mean())
    print(aggr[-1, -1], x[-2:, -2:].mean())

    print("==" * 10)


    agg_blocks(np.random.randn(20, 20), (5,  5))
    agg_blocks(np.random.randn(20, 20), (7,  7))


    x = np.random.randn(5, 5)
    xm = np.ma.masked_where(x < 0, x)
    xm = np.ma.round(xm, decimals=2)

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


    x = np.arange(10)
    print("==" * 10)
    print(x)
    print(agg_blocks(x, block_shape=(3,)))


if __name__ == '__main__':
    pass