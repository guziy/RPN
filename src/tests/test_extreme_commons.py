__author__ = 'huziy'

from datetime import datetime, timedelta
import numpy as np

import pandas as pd

from crcm5.analyse_hdf.return_levels import extreme_commons as ec


def generate_continuous_data():
    np.random.seed(10)
    nt = 5000
    d0 = datetime(1980, 1, 1)
    dt = timedelta(hours=12)
    ts_times = [d0 + i * dt for i in range(nt)]
    ts_vals = np.random.randn(nt).tolist()

    return ts_times, ts_vals


def test_with_cont_data():
    t, v = generate_continuous_data()
    assert ec.is_continuous(ts_times=t), "Continuous data is not recognized as such."

    ec.get_annual_extrema(t, v)


def test_get_annual_extrema():
    pass


def main():
    t, v = generate_continuous_data()

    df = pd.DataFrame(index=t, data=v)

    df_length_per_year = df.groupby(lambda d: d.year).apply(lambda g1: len(g1))

    print(df_length_per_year.head(20))
    print("Time ranges: ", min(t), max(t))
    test_with_cont_data()


if __name__ == '__main__':
    main()