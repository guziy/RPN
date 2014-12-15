import os

__author__ = 'huziy'

import tables as tb
import pandas as pd
from datetime import datetime
import numpy as np


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

    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    with tb.open_file(intf_file) as intf_handle, tb.open_file(no_intf_file) as no_intf_handle:
        # Get runoff tables (level_index == 0 corresponds to runoff from soil)
        traf_table_intf = intf_handle.get_node("/TRAF")
        traf_table_no_intf = no_intf_handle.get_node("/TRAF")

        # Get 2m air temperature in order to distinguish between rain and snow
        swe_table_intf = intf_handle.get_node("/I5")
        swe_table_no_intf = no_intf_handle.get_node("/I5")


        # Get 2m air temperature in order to distinguish between rain and snow
        t2m_table_intf = intf_handle.get_node("/TT")
        t2m_table_no_intf = no_intf_handle.get_node("/TT")

        total_diff = 0
        for year in range(start_year, end_year + 1):
            print "Processing year {}".format(year)

            # Get data for the simulation with interflow
            traf_intf = get_np_arr_sorted_for_year(year, traf_table_intf)
            swe_intf = get_np_arr_sorted_for_year(year, swe_table_intf)
            t2m_intf = get_np_arr_sorted_for_year(year, t2m_table_intf)

            # Get data for the simulation with interflow
            traf_no_intf = get_np_arr_sorted_for_year(year, traf_table_no_intf)
            swe_no_intf = get_np_arr_sorted_for_year(year, swe_table_no_intf)
            t2m_no_intf = get_np_arr_sorted_for_year(year, t2m_table_no_intf)


            place_and_time = (swe_intf > 0) & (t2m_intf > 0) & (swe_no_intf > 0) & (t2m_no_intf > 0) & \
                             (traf_intf > 0) & (traf_no_intf > 0)

            the_diff = np.zeros(place_and_time.shape)

            the_diff[place_and_time] += traf_intf[place_and_time] - traf_no_intf[place_and_time]
            total_diff += the_diff.sum(axis=0)
            print "Number of relevant points: ", np.count_nonzero(place_and_time)
            print "Finished processing {}".format(year)

        # save the figure
        plt.figure()
        total_diff /= float(end_year - start_year + 1)
        plt.pcolormesh(total_diff.transpose())
        plt.colorbar()
        plt.savefig(os.path.join(img_folder, "traf_diff.png"))





    pass


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    params = dict(
        intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
        no_intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=1980, end_year=2010)
    main(**params)