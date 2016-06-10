

# timeseries of lake ice fractions for the Great Lakes, based on the charts from
# the Canadian Ice Service and NOAA National Ice Center.
# the jday corresponds to a day in the 365-day long year
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def get_ts_from_file(path="", start_year=-np.Inf, end_year=np.Inf) -> pd.DataFrame:
    """
    Accept the path to the text files with lake ice fraction timeseries
    :return :
    """

    df = pd.DataFrame.from_csv(path, sep="\s+")

    cnames = df.columns[:]

    for c in cnames:
        y = int(c)
        if y < start_year or y > end_year:
            df.drop(c, axis=1, inplace=True)

    return df


def get_ts_with_real_dates_from_file(path="", start_year=-np.Inf, end_year=np.Inf) -> pd.Series:
    df = get_ts_from_file(path=path, start_year=start_year, end_year=end_year)

    ser_list = []

    jday = df.index
    import calendar

    day = timedelta(days=1)
    for y in range(start_year, end_year + 1):
        offset = 0 if calendar.isleap(y - 1) else 1
        dates = [datetime(y - 1, 1, 1) + day * (jd - offset) for jd in jday]
        s = pd.Series(index=dates, data=df[str(y)].values)
        ser_list.append(s)

    return pd.concat(ser_list)



def main():
    df = get_ts_from_file(path="/RESCUE/skynet3_rech1/huziy/obs_data/Lake_ice_concentration_Great_lakes_timeseries/HUR-30x.TXT",
                          start_year=1980, end_year=1985)


    dfm = df.mean(axis=1)
    plt.figure()
    dfm.plot()
    plt.show()

    for c in df.columns:
        print(c)


if __name__ == '__main__':
    main()