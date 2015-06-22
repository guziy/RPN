import calendar
from collections import OrderedDict
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

__author__ = 'huziy'


class ExtremeProperties(object):
    seed = 10

    # Make it small for testing
    nbootstrap = 1000

    low = "low"
    high = "high"
    extreme_types = [high, low]

    extreme_type_to_return_periods = OrderedDict([
        ("high", [10, 50]),
        ("low", [2, 5]),
    ])

    extreme_type_to_month_of_interest = OrderedDict([
        ("high", range(3, 7)),
        ("low", range(1, 6)),
    ])

    extreme_type_to_n_agv_days = OrderedDict([
        ("high", 1),
        ("low", 15),
    ])

    def __init__(self, ret_lev_dict=None, std_dict=None):
        self.return_lev_dict = ret_lev_dict if ret_lev_dict is not None else {}
        self.std_dict = std_dict if std_dict is not None else {}

    @classmethod
    def get_month_of_interest_for_low_flow(cls):
        return cls.extreme_type_to_month_of_interest[cls.low]

    @classmethod
    def get_month_of_interest_for_high_flow(cls):
        return cls.extreme_type_to_month_of_interest[cls.high]


    def get_low_rl_for_period(self, return_period=2):
        return self.return_lev_dict[self.low][return_period]

    def get_high_rl_for_period(self, return_period=10):
        return self.return_lev_dict[self.high][return_period]

    def get_rl_and_std(self, ex_type=high, return_period=10):
        """
        Return level along with the standard deviation calculated
        using bootstrap
        :param ex_type:
        :param return_period:
        :return:
        """
        return [z[ex_type][return_period] for z in (self.return_lev_dict, self.std_dict)]

    def __str__(self):
        s = ""
        for et in self.extreme_types:
            s += et + ", periods:\n\t{}\n".format(",".join([str(t) for t in self.return_lev_dict[et].keys()]))
        return s



def is_continuous(ts_times=None):
    data_start_date = ts_times[0]
    data_end_date = ts_times[-1]
    nvals = len(ts_times)

    dt_sec = (data_end_date - data_start_date).total_seconds() / (nvals - 1)
    dt = timedelta(seconds=dt_sec)

    for i, t in enumerate(ts_times):

        t1 = data_start_date + i * dt
        if t1 != t:
            return False

        # No need to test more with the given time step
        if i >= 10:
            break

    return True



def get_longest_continuous_interval(full_years):
    # select the longest continuous time interval (considering the full years only)

    """

    :param full_years: sorted list of years
    """

    start_year = full_years[0]
    end_year = full_years[-1]

    # Obviously all years are there
    if end_year - start_year + 1 == len(full_years):
        return start_year, end_year

    assert start_year <= end_year, "The list of years should be sorted!"

    intervals = [[start_year]]
    y_prev = start_year
    for y in full_years:
        if y - y_prev > 1:
            intervals[-1].append(y_prev)
            intervals.append([y, ])

        y_prev = y

    if len(intervals[-1]) == 1:
        intervals[-1].append(full_years[-1])

    durations = [ey - sy + 1 for sy, ey in intervals]
    return intervals[durations.index(max(durations))]


def get_annual_extrema(ts_times=None, ts_vals=None, start_year=1980, end_year=2010):
    """

    :returns DataFrame with low and high columns (the averaging parameters are taken from the static fields of ExtremeProperties)

    :param ts_times: datetime values
    :param ts_vals: values corresponding to ts_times
    :param start_year: start year of the interval of interest
    :param end_year:
    """
    
    # make sure that the values are continuous in time
    # and then select only years without holes
    df = pd.DataFrame(index=ts_times, data=ts_vals)

    df.sort_index(inplace=True)
    
    df = df.select(lambda d: start_year <= d.year <= end_year)
    
    df_recs_per_year = df.groupby(by=lambda d: d.year).apply(lambda g: len(g))

    vals = sorted(np.unique(df_recs_per_year.values), reverse=True)
    n1_per_year, n2_per_year = vals[:2]

    full_years = df_recs_per_year.index[df_recs_per_year.isin([n1_per_year, n2_per_year])]
    print(full_years)

    # Make sure that the year with the biggest number of points is a leap year
    if not all([calendar.isleap(y) for y in df_recs_per_year.index[n1_per_year == df_recs_per_year.index]]):
        raise Exception("The data is not sufficient for the extreme value analysis.")

    # Make sure that the second longest time series hapens during a non-leap year
    if not all([not calendar.isleap(y) for y in df_recs_per_year.index[n2_per_year == df_recs_per_year.index]]):
        raise Exception("The data is not sufficient for the extreme value analysis.")


    start_year, end_year = get_longest_continuous_interval(full_years)

    df = df.select(lambda d: start_year <= d.year <= end_year)

    # Get daily means
    df_daily = df.groupby(by=lambda d: datetime(d.year, d.month, d.day)).mean()


    df_result = {}
    for ex_type in ExtremeProperties.extreme_types:
        # Take the n-day averages
        mask = np.array(range(len(df_daily))) // ExtremeProperties.extreme_type_to_n_agv_days[ex_type]

        print(len(mask), len(df_daily.index[mask]))

        date_to_interval_start = dict(zip(df_daily.index, df_daily.index[mask]))
        months_of_interest = ExtremeProperties.extreme_type_to_month_of_interest[ex_type]
        df_temp = df_daily.groupby(by=lambda d: date_to_interval_start[d]).mean().select(lambda d: d.month in months_of_interest)
        year_groups = df_temp.groupby(lambda d: d.year)
        df_temp = year_groups.max() if ex_type == ExtremeProperties.high else year_groups.min()

        print(ex_type + "-----------------------")

        assert isinstance(df_temp, pd.DataFrame)
        df_result[ex_type] = df_temp
        print(df_temp.head(10))
        # TODO: Fix

    # print(df_result.head())

