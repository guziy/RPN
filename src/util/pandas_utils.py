from datetime import datetime

import pandas as pd

import numpy as np

def get_daily_climatology_from_pandas_series(ts, stamp_dates, years_of_interest=None):
    assert stamp_dates is not None
    assert isinstance(ts, pd.Series)
    assert not isinstance(ts, pd.DataFrame)

    ts_local = ts.resample("D", closed="left").mean().bfill()

    if years_of_interest is not None:
        ts_local = ts_local.select(lambda d: d.year in years_of_interest)

    ts_local = ts_local.select(lambda d: not (d.month == 2 and d.day == 29))

    if len(ts_local) == 0:
        return None, None

    stamp_year = stamp_dates[0].year

    daily_clim = ts_local.groupby(by=lambda the_date: datetime(stamp_year, the_date.month, the_date.day)).mean()

    vals = [daily_clim.ix[d] for d in stamp_dates]

    return stamp_dates, np.array(vals)
