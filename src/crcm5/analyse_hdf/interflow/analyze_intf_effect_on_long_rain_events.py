__author__ = 'huziy'

import tables as tb
from datetime import datetime
import pandas as pd


def get_pandas_panel(rows):
    """

    :param rows: iterable of dicts of the form:
        {
        "year":int, "month": int, "day": int, "hour":int, "minute":int, "second": int,
        "field": 2D numpy array (nx, ny)
        }
    """
    date_keys = ["year", "month", "day", "hour", "minute", "second"]
    return pd.Panel({datetime(*[row[k] for k in date_keys]): pd.DataFrame(row["field"])
                     for row in rows})


def main(intf_file="", no_intf_file="", start_year=1980, end_year=2010):
    """
    Do it on a year by year basis
    :param intf_file:
    :param no_intf_file:
    :param start_year:
    :param end_year:
    """

    # The value below which the precipitation is considered to be 0
    precip_limit_mm_per_day = 0.1
    precip_limit_m_per_s = precip_limit_mm_per_day * (1.0e-3 / (60.0 * 60.0 * 24.0))

    with tb.open_file(intf_file) as intf_handle, tb.open_file(no_intf_file) as no_intf_handle:
        # Get precipitation tables
        pr_table_intf = intf_handle.get_node("/PR")
        pr_table_no_intf = no_intf_handle.get_node("/PR")

        # Get runoff tables (level_index == 0 corresponds to runoff from soil)
        traf_table_intf = intf_handle.get_node("/TRAF")
        traf_table_no_intf = no_intf_handle.get_node("/TRAF")

        # Get 2m air temperature in order to distinguish between rain and snow
        t2m_table_intf = intf_handle.get_node("/TT")
        t2m_table_no_intf = no_intf_handle.get_node("/TT")


        # iterate through all years
        for the_year in range(start_year, end_year + 1):
            # we need level_index == 0 in both cases, so the query will be the same for
            # runoff and precipitation
            query = "(year == {}) & (level_index == 0)".format(the_year)

            pass

    pass


if __name__ == '__main__':
    params = dict(intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
                  no_intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
                  start_year=1980, end_year=2010)
    main(**params)