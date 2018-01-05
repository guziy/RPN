import dask
import dask.array
import numpy as np
import pandas as pd
from numba import jit


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


@jit
def clim_day_percentile_calculator(block, time_sel, missing_value, rolling_mean_window_days=None, percentile=0.5,
                                   start_year=-np.Inf, end_year=np.Inf):

    # return the masked array if all the values for the point are masked
    """

    :param rolling_mean_window_days:
    :param percentile:
    :param time_sel: times corresponding to the first dimension of block
    :param block: 3D field (nt, nx, ny)
    :param missing_value:
    :return:
    """

    print("My block shape: {}".format(block.shape))

    print("missing value inside of the block calculator: {}".format(missing_value))

    assert 0 <= percentile <= 1
    new_shape = (365,) + block.shape[1:]

    first_field = block[0]

    # check if maybe all values are missing
    if np.isnan(missing_value):
        if np.all(np.isnan(first_field)):
            return missing_value * dask.array.ones(new_shape, chunks=new_shape)
    elif np.all(np.less(np.abs(first_field - missing_value), 1e-5)):
        return missing_value * dask.array.ones(new_shape, chunks=new_shape)


    mask = np.isnan(first_field)
    if hasattr(first_field, "mask"):
        mask = mask | first_field.mask

    ix, jy = np.where(~mask)
    good_data = block[:, ix, jy]

    s = pd.DataFrame(data=good_data, index=time_sel)

    s = s[(~((s.index.month == 2) & (s.index.day == 29))) & (start_year <= s.index.year) & (s.index.year <= end_year)]
    assert isinstance(s, pd.DataFrame)

    if rolling_mean_window_days is not None:
        s = s.rolling(rolling_mean_window_days, center=True).mean().bfill().ffill()


    # Each group is a dataframe with the rows(axis=0) for a day of different years
    grouped = s.groupby([s.index.month, s.index.day])
    daily_perc = grouped.quantile(q=percentile)
    print("computed daily percentile")


    result = np.ma.masked_all(new_shape)
    result[:, ix, jy] = daily_perc.values



    return result  # <- Should be (365, nx, ny)