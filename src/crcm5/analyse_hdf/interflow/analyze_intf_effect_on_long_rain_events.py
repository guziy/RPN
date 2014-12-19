
import matplotlib
matplotlib.use("Agg")

import os
import pickle
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

__author__ = 'huziy'

import tables as tb
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import crcm5.analyse_hdf.do_analysis_using_pytables as analysis

from mpl_toolkits.basemap import maskoceans
import ctypes

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


def get_longest_rain_event_durations_from_tables(pr, traf=None, t2m=None, pr_lower_lim=0.0):
    """
    Get maximum numbers of time steps where
    :param pr:
    :param traf:
    :param t2m: 2m air temperature in Celsius
    """

    # durations of the longest and current rain events
    durations_max = np.zeros(pr.shape[1:])
    durations_current = np.zeros(pr.shape[1:])

    acc_runoff_max = np.zeros(pr.shape[1:])
    acc_runoff_current = np.zeros(pr.shape[1:])

    # loop in time
    for t in range(pr.shape[0]):
        p = pr[t]
        r = traf[t]

        # Use 2m air temperature to distinguish between rain and snow (In my opinion this is more correct)
        t2m_field = t2m[t]
        is_it_raining_now = (p >= pr_lower_lim) & (t2m_field > 0)

        durations_current[is_it_raining_now] += 1
        acc_runoff_current[is_it_raining_now] += r[is_it_raining_now]

        update_duration_max = (~is_it_raining_now) & (durations_max < durations_current)

        # If the maximum duration should be updated somewhere
        if np.any(update_duration_max):
            durations_max[update_duration_max] = durations_current[update_duration_max]
            acc_runoff_max[update_duration_max] = acc_runoff_current[update_duration_max]

        durations_current[~is_it_raining_now] = 0
        acc_runoff_current[~is_it_raining_now] = 0

    return durations_max, acc_runoff_max


def get_longest_rain_durations_for_files(intf_file="", no_intf_file="",
                                         start_year=1980,
                                         end_year=2010):
    # The value below which the precipitation is considered to be 0
    """
    TODO: add caching
    :param intf_file:
    :param no_intf_file:
    :param start_year:
    :param end_year:
    """

    precip_lower_limit_mm_per_day = 20

    key_str = intf_file + no_intf_file + "{}".format(precip_lower_limit_mm_per_day)
    cache_file = "{}_{}-{}.long-events-intf.cache".format(
        ctypes.c_size_t(hash(key_str)).value,
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
        total_acc_runoff_nointf = 0
        total_acc_runoff_intf = 0
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
            print pr_no_intf_panel

            print "With intf panels: "
            print pr_intf_panel

            # Get durations of the longest events during the year when no interflow is present
            max_durations_nointf, acc_runoff_nointf = get_longest_rain_event_durations_from_tables(
                pr_no_intf_panel.values,
                traf=traf_no_intf_panel.values,
                t2m=t2m_no_intf_panel.values,
                pr_lower_lim=precip_lower_limit_m_per_s,
            )
            no_intf_all_max_durations.append(max_durations_nointf)

            # Get durations of the longest events during the year when interflow is present
            max_durations_intf, acc_runoff_intf = get_longest_rain_event_durations_from_tables(
                pr_intf_panel.values,
                traf=traf_intf_panel.values,
                t2m=t2m_intf_panel.values,
                pr_lower_lim=precip_lower_limit_m_per_s,
            )
            intf_all_max_durations.append(max_durations_intf)

            total_acc_runoff_intf += acc_runoff_intf
            total_acc_runoff_nointf += acc_runoff_nointf

            print "Finished processing year: {}".format(the_year)

    nyears = float(end_year - start_year + 1)
    total_acc_runoff_nointf /= nyears
    total_acc_runoff_intf /= nyears

    pickle.dump([no_intf_all_max_durations,
                 total_acc_runoff_nointf,
                 intf_all_max_durations,
                 total_acc_runoff_intf], open(cache_file, "w"))
    return no_intf_all_max_durations, total_acc_runoff_nointf, intf_all_max_durations, total_acc_runoff_intf


def get_nevents_for_each_duration(durations, minlength=0):
    durmax = max(durations.max(), minlength)
    return np.asarray([np.count_nonzero(dt == durations) for dt in range(1, durmax + 1)])


def plot_nevents_duration_curves(no_intf_max_durations, intf_max_durations,
                                 img_path="test.png", dt_hours=3):
    """
    dt_hours  - output step in hours

    """

    plt.figure()

    minlength = max(no_intf_max_durations.max(), intf_max_durations.max()) + 5
    print minlength

    max_duration_limit = 10  # output steps

    nevents_intf = get_nevents_for_each_duration(intf_max_durations, minlength=minlength)
    nevents_no_intf = get_nevents_for_each_duration(no_intf_max_durations, minlength=minlength)

    delta = nevents_intf - nevents_no_intf

    durations = dt_hours * np.array(range(1, minlength + 1))
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


def plot_surface_runoff_differences(x, y, basemap, mask, no_intf_acc_runoff, intf_acc_runoff,
                                    img_path="roff_due_to_intf.png", dt_hours=3):

    fig = plt.figure()

    assert isinstance(fig, Figure)
    print fig.get_figwidth()
    fig.set_size_inches(fig.get_figwidth() * 3, fig.get_figheight() * 1.4)

    gs = GridSpec(1, 5, width_ratios=[1, 1, 0.05, 1.4, 0.05])

    ax_list = []

    clevs = [0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40]
    cmap_field = cm.get_cmap("rainbow", len(clevs) - 1)
    bn = BoundaryNorm(clevs, len(clevs) - 1)


    no_intf_acc_runoff_mm = np.ma.masked_where(mask, no_intf_acc_runoff) * dt_hours * 60 * 60
    intf_acc_runoff_mm = np.ma.masked_where(mask, intf_acc_runoff) * dt_hours * 60 * 60

    vmax = clevs[-1]  # mm

    # Plot values
    ax = fig.add_subplot(gs[0, 0])
    basemap.pcolormesh(x, y, no_intf_acc_runoff_mm, vmin=0, vmax=vmax, cmap=cmap_field, norm=bn)
    ax.set_title("No intf., surface runoff")
    ax_list.append(ax)

    ax = fig.add_subplot(gs[0, 1])
    im = basemap.pcolormesh(x, y, intf_acc_runoff_mm, vmin=0, vmax=vmax, cmap=cmap_field, norm=bn)
    ax.set_title("Intf., surface runoff")
    ax_list.append(ax)

    cax = fig.add_subplot(gs[0, 2])
    cb = plt.colorbar(im, cax=cax, ticks=clevs)
    cb.ax.set_xlabel("mm")

    # Plot differences
    clevs = [0.5, 1, 5, 30, 100, 150]
    clevs = [-c for c in reversed(clevs)] + clevs
    cmap_diff = cm.get_cmap("bwr", len(clevs) - 1)
    bn = BoundaryNorm(clevs, len(clevs) - 1)

    ax = fig.add_subplot(gs[0, 3])
    diff = intf_acc_runoff_mm - no_intf_acc_runoff_mm
    im = basemap.pcolormesh(x, y, diff, cmap=cmap_diff, norm=bn)
    ax.set_title(r"$R_{\rm intf.} - R_{\rm nointf.}$; " +
                 r"$\sum\Delta_{i, j}$ = " +
                 "{:.2f} ".format(diff.sum() * 100.0 * 1.0e-6) +
                 r"${\rm km^3 / year}$" + "\n")
    ax_list.append(ax)

    cax = fig.add_subplot(gs[0, 4])
    cb = plt.colorbar(im, cax=cax, ticks=clevs)
    cb.ax.set_xlabel("mm")

    # Draw coastlines
    for ax in ax_list:
        basemap.drawcoastlines(ax=ax)

    fig.savefig(img_path)


def main(intf_file="", no_intf_file="", start_year=1980, end_year=2010, dt_hours=3):
    """
    Do it on a year by year basis
    :param intf_file:
    :param no_intf_file:
    :param start_year:
    :param end_year:

    """
    matplotlib.rc("font", size=20)
    img_folder = "long-rain-events-30y"
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    # Calculate the durations of the longest rain events in both simulations
    no_intf_all_max_durations, no_intf_acc_runoff, intf_all_max_durations, intf_acc_runoff = \
        get_longest_rain_durations_for_files(
            intf_file=intf_file, no_intf_file=no_intf_file, start_year=start_year, end_year=end_year)

    # Debug: visualize
    cmap = cm.get_cmap("rainbow", 20)

    lons, lats, basemap = analysis.get_basemap_from_hdf(file_path=no_intf_file)
    x, y = basemap(lons, lats)

    plt.figure()
    mean_max_durations_nointf = np.mean(no_intf_all_max_durations, axis=0).astype(int)
    im = basemap.pcolormesh(x, y, mean_max_durations_nointf, vmin=0, vmax=50, cmap=cmap)
    basemap.drawcoastlines()
    plt.title("no - intf")
    plt.colorbar(im)
    print mean_max_durations_nointf.min(), mean_max_durations_nointf.max(), mean_max_durations_nointf.mean()
    plt.savefig(os.path.join(img_folder, "no-intf-durations.png"))

    plt.figure()
    mean_max_durations_intf = np.mean(intf_all_max_durations, axis=0).astype(int)
    im = basemap.pcolormesh(x, y, mean_max_durations_intf, vmin=0, vmax=50, cmap=cmap)
    basemap.drawcoastlines()
    plt.title("intf")
    plt.colorbar(im)
    print mean_max_durations_intf.min(), mean_max_durations_intf.max(), mean_max_durations_intf.mean()
    plt.savefig(os.path.join(img_folder, "intf-durations.png"))

    # Plot the interflow effect on the longest rain events
    mask = maskoceans(lons, lats, mean_max_durations_intf, inlands=True).mask
    plt.figure()
    clevs = [0.5, 1, 5, 30, 100, 150]
    clevs = [-c for c in reversed(clevs)] + clevs
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    cmap_diff = cm.get_cmap("bwr", len(clevs) - 1)
    diff = np.ma.masked_where(mask, (mean_max_durations_intf - mean_max_durations_nointf) * dt_hours)

    im = basemap.pcolormesh(x, y, diff, cmap=cmap_diff, vmin=clevs[0], vmax=clevs[-1], norm=bn)
    basemap.drawcoastlines()
    plt.title("intf - nointf" + r", $\sum\Delta_{i, j}$ = " + "{}\n".format(diff.sum()))
    cb = plt.colorbar(im)
    cb.ax.set_title("hours")
    plt.savefig(os.path.join(img_folder, "diff_intf-nointf_durations.png"))

    # Plot differences in surface runoff
    plot_surface_runoff_differences(x, y, basemap, mask, no_intf_acc_runoff, intf_acc_runoff, dt_hours=dt_hours,
                                    img_path=os.path.join(img_folder, "runoff_during_long_rain_events.png"))

    # Plot numbers of events of different durations
    plot_nevents_duration_curves(mean_max_durations_nointf[~mask], mean_max_durations_intf[~mask],
                                 img_path=os.path.join(img_folder, "nevents_vs_duration_over_land.png"))

    plot_nevents_duration_curves(mean_max_durations_nointf[mask], mean_max_durations_intf[mask],
                                 img_path=os.path.join(img_folder, "nevents_vs_duration_over_ocean-and-lakes.png"))


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    params = dict(
        intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
        no_intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=1980, end_year=2010)
    main(**params)