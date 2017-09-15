

import numpy as np
import pandas as pd
from numba import jit





def agg_blocks_skimage_improved(data, block_shape: tuple, func=np.nanmean, pad_value=np.nan):
    """
    Modified from skimage.measure.block_reduce
    :param data:
    :param block_shape:
    :param func:
    :param pad_value:
    """
    from skimage.util import view_as_blocks

    assert data.ndim == len(block_shape), "Block should have the same number of dimensions as the input array(ndim={}).".format(data.ndim)


    if data.ndim == 1:
        data.shape = data.shape + (1, )
        block_shape = block_shape + (1, )


    pad_width = []
    for i in range(data.ndim):
        if block_shape[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an "
                             "image.")
        if data.shape[i] % block_shape[i] != 0:
            after_width = block_shape[i] - (data.shape[i] % block_shape[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))

    image = np.pad(data, pad_width=pad_width, mode='constant', constant_values=pad_value)

    out = view_as_blocks(image, block_shape)
    result_shape = out.shape[:-2]
    out = out.reshape((-1, ) + block_shape)

    @jit(nopython=True)
    def wrap(*args):
        return func(*args)


    return np.array([wrap(chunk) for chunk in out]).reshape(result_shape).squeeze()






def agg_blocks_gridtools(data, block_shape: tuple):
    from gridtools import resampling


    if data.ndim == 1:
        data.shape = data.shape + (1, )
        block_shape = block_shape + (1, )

    new_shape = [data.shape[j] // block_shape[j] + int(data.shape[j] % block_shape[j] != 0) for j in range(data.ndim)]
    new_shape = [max(s, 1) for s in new_shape]

    return resampling.resample_2d(data, new_shape[0], new_shape[1], ds_method=resampling.DS_MEAN)



def agg_blocks_dask(data, block_shape: tuple, func=np.mean, num_workers=10):
    from dask import array as d_arr

    if hasattr(data, "mask"):
        data_ = np.where(~data.mask, data, np.nan)
    else:
        data_ = data

    if isinstance(data, d_arr.Array):
        arr = data.rechunk(block_shape)
    else:
        arr = d_arr.from_array(data_, chunks=block_shape)

    def wrap(*args, **kwargs):
        res = func(*args, **kwargs)
        a_shape = data_.ndim * (None, )
        return res[a_shape]


    return arr.map_blocks(wrap, dtype=np.float32).compute(num_workers=num_workers)



def agg_blocks(data, block_shape: tuple, func=np.mean, indices=None):

    assert data.ndim == len(block_shape)

    if indices is None:
        indices = np.indices(data.shape)


    new_shape = []
    for j in range(len(block_shape)):
        new_shape.append(data.shape[j] // block_shape[j] + int(data.shape[j] % block_shape[j] != 0))

    new_data = np.ma.masked_all(new_shape)

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

    # aggregation: group and reduce
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
#    print(x)

    aggr = np.round(agg_blocks(x, (2,  2)), decimals=2)
#    print(aggr)


#    print(aggr[0, 0], x[:2, :2].mean())
#    print(aggr[-1, -1], x[-2:, -2:].mean())

    print("==" * 10)


    agg_blocks(np.random.randn(20, 20), (5,  5))
    agg_blocks(np.random.randn(20, 20), (7,  7))


    x = np.random.randn(5, 5)
    xm = np.ma.masked_where(x < 0, x)
    xm = np.ma.round(xm, decimals=2)

    def __agg(arr):
        if isinstance(arr, pd.Series):
            v = arr.values
        else:
            v = arr

        # print(v.shape, v)

        bad = np.isnan(v)
        if hasattr(v, "mask"):
            bad = v.mask | bad


        if not np.all(bad):
            return v[~bad].mean()
        else:
            __x = np.nan * np.zeros((1, ))

            return __x

    agg1 = agg_blocks(xm, (2, 2), func=__agg)
    print(xm)
    print(agg1)
    print("dask")
    print(agg_blocks_dask(xm, (2, 2), func=__agg))

    print("skimage-improved")
    xnan = xm.copy()
    xnan[xm.mask] = np.nan
    print(xnan)
    print(agg_blocks_skimage_improved(xnan, (2, 2)))

    print("gridtools")
    print(xm)
    print(agg_blocks_gridtools(xm, (2, 2)))



    x = np.arange(10)
    print("==" * 10 + "- pandas")
    print(x)
    print(agg_blocks(x, block_shape=(3,)))

    print("==" * 10 + "- dask")
    print(x)
    # print(agg_blocks_gridtools(x, block_shape=(2,)))
    print(agg_blocks_dask(x, block_shape=(3, )))


    print("==" * 10 + "- skimage improved")
    print(x)
    # print(agg_blocks_gridtools(x, block_shape=(2,)))
    print(agg_blocks_skimage_improved(x, block_shape=(3,)))





if __name__ == '__main__':
    pass