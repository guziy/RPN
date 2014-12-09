__author__ = 'huziy'

import tables as tb
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Calculates durations in time steps of the longest rain events for each grid point,
# and then compare them for simulations with and without interflow
#


def get_pandas_panel(rows):
    """

    :param rows: iterable of dicts of the form:
        {
        "year":int, "month": int, "day": int, "hour":int, "minute":int, "second": int,
        "field": 2D numpy array (nx, ny)
        }
    :return pd.Panel
    """
    date_keys = ["year", "month", "day", "hour", "minute", "second"]
    return pd.Panel({datetime(*[row[k] for k in date_keys]): pd.DataFrame(row["field"])
                     for row in rows})


def get_longest_rain_event_durations(pr, traf, pr_lower_lim=0.0, traf_lower_lim=0.0):
    """
    Get maximum numbers of time steps where
    :param pr:
    :param traf:
    """

    # durations of the longest and current rain events
    durations_max = np.zeros(pr.shape[1:])
    durations_current = np.zeros(pr.shape[1:])

    # loop in time
    for t in range(pr.shape[0]):
        p = pr[t]
        r = traf[t]

        is_it_raining_now = (p >= pr_lower_lim) & (r >= traf_lower_lim)

        durations_current[is_it_raining_now] += 1
        update_duration_max = (~is_it_raining_now) & (durations_max < durations_current)

        # If the maximum duration should be updated somewhere
        if np.any(update_duration_max):
            durations_max[update_duration_max] = durations_current[update_duration_max]

        durations_current[~is_it_raining_now] = 0

    return durations_max

    pass


def main(intf_file="", no_intf_file="", start_year=1980, end_year=2010):
    """
    Do it on a year by year basis
    :param intf_file:
    :param no_intf_file:
    :param start_year:
    :param end_year:

    """

    # The value below which the precipitation is considered to be 0
    precip_lower_limit_mm_per_day = 0.1
    precip_lower_limit_m_per_s = precip_lower_limit_mm_per_day * (1.0e-3 / (60.0 * 60.0 * 24.0))

    # /10.0 so it detects smaller surface runoffs
    traf_lower_limit_mm_per_s = precip_lower_limit_m_per_s / 10.0 * 1.0e3


    # The result array
    traf_diff_composite = None


    # Durations of precip events
    durations_ntime_steps = None

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

        assert isinstance(pr_table_no_intf, tb.Table)

        # iterate through all years
        for the_year in range(start_year, end_year + 1):
            print "Start processing year: {}".format(the_year)
            # we need level_index == 0 in both cases, so the query will be the same for
            # runoff and precipitation
            query = "(year == {}) & (level_index == 0)".format(the_year)


            # Small test -- Delete later

            ysum = 0
            for y in range(1980, 1991):
                ysum1 = len(traf_table_intf.get_where_list("(year == {}) & (level_index == 0)".format(y)))
                print y, ysum1
                ysum += ysum1

                ysum1 = len(traf_table_intf.get_where_list("(year == {}) & (level_index == 1)".format(y)))
                print "level_index = 1", ysum1

                ysum1 = len(traf_table_intf.get_where_list("(year == {}) & (level_index == 2)".format(y)))
                print "level_index = 2", ysum1

                ysum1 = len(traf_table_intf.get_where_list("(year == {}) & (level_index == 3)".format(y)))
                print "level_index = 3", ysum1

                ysum1 = len(traf_table_intf.get_where_list("(year == {}) & (level_index == 4)".format(y)))
                print "level_index = 4", ysum1

            print "All years sum (for level_index = 0): ", ysum

            coords = traf_table_intf.get_where_list("(year == 1980) & (month == 3) & (day == 2) & (hour == 18)")
            print coords
            for row in traf_table_intf.read_coordinates(coords):
                print row["level_index"]


            # Get data for a year into a pandas Panel.
            coords = pr_table_no_intf.get_where_list(query)
            pr_no_intf_panel = get_pandas_panel(pr_table_no_intf.read_coordinates(coords))
            print len(coords)

            coords = pr_table_intf.get_where_list(query)
            pr_intf_panel = get_pandas_panel(pr_table_intf.read_coordinates(coords))
            print len(coords)

            coords = traf_table_no_intf.get_where_list(query)
            traf_no_intf_panel = get_pandas_panel(traf_table_no_intf.read_coordinates(coords))
            print len(coords)

            coords = traf_table_intf.get_where_list(query)
            print len(coords)
            traf_intf_panel = get_pandas_panel(traf_table_intf.read_coordinates(coords))

            # Sort by date
            pr_no_intf_panel = pr_no_intf_panel.sort_index(axis="items")
            pr_intf_panel = pr_intf_panel.sort_index(axis="items")

            traf_no_intf_panel = traf_no_intf_panel.sort_index(axis="items")
            traf_intf_panel = traf_intf_panel.sort_index(axis="items")

            print "No intf panels: "
            print traf_no_intf_panel
            print pr_no_intf_panel

            print "With intf panels: "
            print traf_intf_panel
            print pr_intf_panel

            print "Missing dates detection: ------------------"
            found_missing_dates = 0
            for d in traf_no_intf_panel.items:
                if d not in traf_intf_panel.items:
                    print d
                    found_missing_dates += 1

            print "---------------------------------"
            print "Found {} missing dates!".format(found_missing_dates)

            max_durations_nointf = get_longest_rain_event_durations(pr_no_intf_panel.values,
                                                                    traf_no_intf_panel.values,
                                                                    pr_lower_lim=precip_lower_limit_m_per_s,
                                                                    traf_lower_lim=traf_lower_limit_mm_per_s)

            max_durations_intf = get_longest_rain_event_durations(pr_intf_panel.values,
                                                                  traf_intf_panel.values,
                                                                  pr_lower_lim=precip_lower_limit_m_per_s,
                                                                  traf_lower_lim=traf_lower_limit_mm_per_s)


            # Debug: visualize
            plt.figure()
            im = plt.pcolormesh(max_durations_nointf.transpose())
            plt.title("no - intf")
            plt.colorbar(im)

            plt.figure()
            im = plt.pcolormesh(max_durations_intf.transpose())
            plt.title("intf")
            plt.colorbar(im)


            # Plot the interflow effect on the longest rain events
            plt.figure()
            diff = max_durations_intf - max_durations_nointf
            im.pcolormesh(diff.transpose())
            plt.title("intf - nointf")
            plt.colorbar(im)

            plt.show()

            print pr_no_intf_panel

            print "Finished processing year: {}".format(the_year)

    pass


if __name__ == '__main__':
    params = dict(
        intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5",
        no_intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=1980, end_year=1980)
    main(**params)