import ctypes
import os
import pickle
from matplotlib.colors import BoundaryNorm

__author__ = 'huziy'

import tables as tb
import pandas as pd
from datetime import datetime
import numpy as np

# To plot images and do not care about the DISPLAY variable
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

import matplotlib.pyplot as plt


def get_pandas_panel_sorted_for_year(year, the_table):
    assert isinstance(the_table, tb.Table)

    query = "(year == {}) & (level_index == 0)".format(year)

    coords = the_table.get_where_list(query)
    rows = the_table.read_coordinates(coords)

    date_keys = ["year", "month", "day", "hour", "minute", "second"]
    return pd.Panel({datetime(*[row[k] for k in date_keys]): pd.DataFrame(row["field"])
                     for row in rows})


def get_np_arr_sorted_for_year(year, the_table):
    return get_pandas_panel_sorted_for_year(year, the_table).values



def get_runoff_diff_composit(swe_table_intf=None, traf_table_intf=None,
                             swe_table_no_intf=None, traf_table_no_intf=None,
                             start_year=1980, end_year=2010):

    total_diff = None

    for year in range(start_year, end_year + 1):
        print "Processing year {}".format(year)

        # Get data for the simulation with interflow
        traf_intf = get_np_arr_sorted_for_year(year, traf_table_intf)
        swe_intf = get_np_arr_sorted_for_year(year, swe_table_intf)

        # Get data for the simulation with interflow
        traf_no_intf = get_np_arr_sorted_for_year(year, traf_table_no_intf)
        swe_no_intf = get_np_arr_sorted_for_year(year, swe_table_no_intf)

        nt, nx, ny = swe_intf.shape
        if total_diff is None:
            total_diff = np.zeros(swe_intf.shape[1:])

        for t in range(1, nt):
            place_where_melts = (swe_intf[t] < swe_intf[t - 1]) & (swe_no_intf[t] < swe_no_intf[t - 1])

            total_diff[place_where_melts] += traf_intf[t][place_where_melts] - traf_no_intf[t][place_where_melts]

        # print "Number of relevant points: ", np.count_nonzero(the_diff)
        print "Finished processing {}".format(year)
    return total_diff


def main(intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
         no_intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
         start_year=1980, end_year=2010):
    """
    Get runoff difference only for melting periods
    :param intf_file:
    :param no_intf_file:
    :param start_year:
    :param end_year:
    """
    img_folder = "intf_during_melting"

    cache_file = "intf_during_melting_{}_{}-{}.cache".format(ctypes.c_size_t(hash(intf_file + no_intf_file)).value,
                                                             start_year, end_year)

    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    with tb.open_file(intf_file) as intf_handle, tb.open_file(no_intf_file) as no_intf_handle:
        # Get runoff tables (level_index == 0 corresponds to runoff from soil)
        traf_table_intf = intf_handle.get_node("/TRAF")
        traf_table_no_intf = no_intf_handle.get_node("/TRAF")

        # Get swe to determine when melting is happening
        swe_table_intf = intf_handle.get_node("/I5")
        swe_table_no_intf = no_intf_handle.get_node("/I5")

        if os.path.isfile(cache_file):
            total_diff = pickle.load(open(cache_file))
        else:
            total_diff = get_runoff_diff_composit(swe_table_intf=swe_table_intf,
                                                  traf_table_intf=traf_table_intf,
                                                  swe_table_no_intf=swe_table_no_intf,
                                                  traf_table_no_intf=traf_table_no_intf,
                                                  start_year=start_year, end_year=end_year)

        # save the figure
        plt.figure()
        total_diff /= float(end_year - start_year + 1)

        # Convert to mm
        total_diff *= 3 * 60 * 60
        lons, lats, bm = analysis.get_basemap_from_hdf(intf_file)

        x, y = bm(lons, lats)

        clevs = [0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 100, 150, 200]
        clevs = [-c for c in reversed(clevs)] + clevs

        cmap = cm.get_cmap("bwr", len(clevs) - 1)
        bn = BoundaryNorm(clevs, len(clevs) - 1)
        im = bm.pcolormesh(x, y, total_diff, cmap=cmap, norm=bn)
        bm.colorbar(im, ticks=clevs)
        plt.savefig(os.path.join(img_folder, "traf_diff_{}-{}.png".format(start_year, end_year)))

    pass


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    params = dict(
        intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
        no_intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=1980, end_year=2010)
    main(**params)