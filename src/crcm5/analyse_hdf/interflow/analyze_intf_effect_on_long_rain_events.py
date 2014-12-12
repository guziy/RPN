import os
import pickle
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

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


def get_longest_rain_event_durations_from_tables(pr, traf, pr_lower_lim=0.0, traf_lower_lim=0.0):
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


def get_longest_rain_durations_for_files(intf_file="", no_intf_file="", start_year=1980,
                                         end_year=2010):
    # The value below which the precipitation is considered to be 0
    """
    TODO: add caching
    :param intf_file:
    :param no_intf_file:
    :param start_year:
    :param end_year:
    """

    precip_lower_limit_mm_per_day = 0.5

    cache_file = "{}_{}-{}.long-events-intf.cache".format(
        hash(intf_file + no_intf_file + "{}".format(precip_lower_limit_mm_per_day)),
        start_year, end_year)

    if os.path.isfile(cache_file):
        return pickle.load(open(cache_file))

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
        no_intf_all_max_durations = []
        intf_all_max_durations = []
        for the_year in range(start_year, end_year + 1):
            print "Start processing year: {}".format(the_year)
            # we need level_index == 0 in both cases, so the query will be the same for
            # runoff and precipitation
            query = "(year == {}) & (level_index == 0)".format(the_year)

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
            traf_intf_panel = get_pandas_panel(traf_table_intf.read_coordinates(coords))
            print len(coords)

            coords = t2m_table_no_intf.get_where_list(query)
            t2m_no_intf_panel = get_pandas_panel(t2m_table_no_intf.read_coordinates(coords))
            print len(coords)

            coords = t2m_table_intf.get_where_list(query)
            t2m_intf_panel = get_pandas_panel(t2m_table_intf.read_coordinates(coords))
            print len(coords)

            # Sort by date
            pr_no_intf_panel = pr_no_intf_panel.sort_index(axis="items")
            pr_intf_panel = pr_intf_panel.sort_index(axis="items")

            traf_no_intf_panel = traf_no_intf_panel.sort_index(axis="items")
            traf_intf_panel = traf_intf_panel.sort_index(axis="items")

            t2m_no_intf_panel = t2m_no_intf_panel.sort_index(axis="items")
            t2m_intf_panel = t2m_intf_panel.sort_index(axis="items")

            print "No intf panels: "
            print traf_no_intf_panel
            print pr_no_intf_panel

            print "With intf panels: "
            print traf_intf_panel
            print pr_intf_panel


            # Get durations of the longest events during the year when no interflow is present
            max_durations_nointf = get_longest_rain_event_durations_from_tables(pr_no_intf_panel.values,
                                                                                traf_no_intf_panel.values,
                                                                                pr_lower_lim=precip_lower_limit_m_per_s,
                                                                                traf_lower_lim=traf_lower_limit_mm_per_s)
            no_intf_all_max_durations.append(max_durations_nointf)

            # Get durations of the longest events during the year when interflow is present
            max_durations_intf = get_longest_rain_event_durations_from_tables(pr_intf_panel.values,
                                                                              traf_intf_panel.values,
                                                                              pr_lower_lim=precip_lower_limit_m_per_s,
                                                                              traf_lower_lim=traf_lower_limit_mm_per_s)
            intf_all_max_durations.append(max_durations_intf)

            print "Finished processing year: {}".format(the_year)

    pickle.dump([no_intf_all_max_durations, intf_all_max_durations], open(cache_file, "w"))
    return no_intf_all_max_durations, intf_all_max_durations



def plot_nevents_duration_curves(no_intf_max_durations, intf_max_durations,
                                 img_path="test.png", dt_hours=3):
    """
    dt_hours  - output step in hours

    """

    plt.figure()


    nevents_intf = np.bincount(intf_max_durations.flatten(), minlength=50)[1:]
    nevents_no_intf = np.bincount(no_intf_max_durations.flatten(), minlength=50)[1:]
    delta = nevents_intf - nevents_no_intf

    durations = dt_hours * np.array(range(1, 50))
    plt.plot(durations, nevents_intf, label="(Intf.)", lw=2)
    plt.plot(durations, nevents_no_intf, label="(No Intf.)", lw=2)
    plt.plot(durations, delta, "--", label="(Intf.) - (No Intf.)", lw=2)

    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [0] * 2, "k")

    plt.legend()

    plt.ylabel("N points")
    plt.xlabel("Duration (hours)")

    plt.savefig(img_path)


    pass


def main(intf_file="", no_intf_file="", start_year=1980, end_year=2010):
    """
    Do it on a year by year basis
    :param intf_file:
    :param no_intf_file:
    :param start_year:
    :param end_year:

    """
    img_folder = "long-rain-events-30y"
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)


    # Calculate teh durations of the longest rain events in both simulations
    no_intf_all_max_durations, intf_all_max_durations = get_longest_rain_durations_for_files(
        intf_file=intf_file, no_intf_file=no_intf_file, start_year=start_year, end_year=end_year
    )


    # Debug: visualize
    cmap = cm.get_cmap("rainbow", 20)

    plt.figure()
    mean_max_durations_nointf = np.mean(no_intf_all_max_durations, axis=0).astype(int)
    im = plt.pcolormesh(mean_max_durations_nointf.transpose(), vmin=0, vmax=50, cmap=cmap)
    plt.title("no - intf")
    plt.colorbar(im)
    print mean_max_durations_nointf.min(), mean_max_durations_nointf.max(), mean_max_durations_nointf.mean()
    plt.savefig(os.path.join(img_folder, "no-intf-durations.png"))

    plt.figure()
    mean_max_durations_intf = np.mean(intf_all_max_durations, axis=0).astype(int)
    im = plt.pcolormesh(mean_max_durations_intf.transpose(), vmin=0, vmax=50, cmap=cmap)
    plt.title("intf")
    plt.colorbar(im)
    print mean_max_durations_intf.min(), mean_max_durations_intf.max(), mean_max_durations_intf.mean()
    plt.savefig(os.path.join(img_folder, "intf-durations.png"))


    # Plot the interflow effect on the longest rain events

    plt.figure()
    clevs = [1, 2, 4, 6, 8, 10, 15]
    clevs = [-c for c in reversed(clevs)] + clevs
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    cmap_diff = cm.get_cmap("bwr", len(clevs) - 1)
    diff = mean_max_durations_intf - mean_max_durations_nointf
    im = plt.pcolormesh(diff.transpose(), cmap=cmap_diff, vmin=-10, vmax=10, norm=bn)
    plt.title("intf - nointf")
    plt.colorbar(im)
    plt.savefig(os.path.join(img_folder, "diff_intf-nointf_durations.png"))



    # Plot numbers of events of different durations
    plot_nevents_duration_curves(mean_max_durations_nointf, mean_max_durations_intf,
                                 img_path=os.path.join(img_folder, "nevents_vs_duration.png"))




    pass


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    params = dict(
        intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
        no_intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=1980, end_year=1980)
    main(**params)