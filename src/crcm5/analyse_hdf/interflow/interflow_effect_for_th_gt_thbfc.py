from matplotlib import cm
from matplotlib.colors import BoundaryNorm

__author__ = 'huziy'

import os
import ctypes
import tables as tb
import pickle
import matplotlib.pyplot as plt
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
from rpn.rpn import RPN
import numpy as np


def get_bulk_field_capacity(path="/skynet3_rech1/huziy/geofields_interflow_exp/pm1979010100_00000000p"):
    r = RPN(path)
    data = r.get_first_record_for_name("D9")
    r.close()
    return data


def get_runoff_differences_composit(traf_table_intf=None, th_table_intf=None,
                                    traf_table_no_intf=None, th_table_no_intf=None, thbfc_field=None,
                                    start_year=None, end_year=None, dt=3 * 3600):
    """

    :param traf_table_intf:
    :param th_table_intf:
    :param traf_table_no_intf:
    :param th_table_no_intf:
    :param thbfc_field:
    :param start_year:
    :param end_year:
    :param dt:
    :return: mean annual surface runoff differences where bulk field capacity is smaller than the soil misture
    """
    level_index = 0

    total_diff = None
    for y in range(start_year, end_year + 1):
        th_no_intf = analysis.get_np_arr_sorted_for_year(y, th_table_no_intf, level_index=level_index)
        traf_no_intf = analysis.get_np_arr_sorted_for_year(y, traf_table_no_intf, level_index=level_index)

        th_intf = analysis.get_np_arr_sorted_for_year(y, th_table_intf, level_index=level_index)
        traf_intf = analysis.get_np_arr_sorted_for_year(y, traf_table_intf, level_index=level_index)


        if total_diff is None:
            total_diff = np.zeros(th_no_intf.shape[1:])


        for t in range(th_no_intf.shape[0]):
            th1 = th_no_intf[t]
            th2 = th_intf[t]

            r1 = traf_no_intf[t]
            r2 = traf_intf[t]

            cond = (th1 > thbfc_field) & (th2 > thbfc_field)
            total_diff[cond] += (r2 - r1)[cond]

        print "Finished processing : {} from {}-{}".format(y, start_year, end_year)

    total_diff *= dt
    total_diff /= float(end_year - start_year + 1)
    return total_diff



def main(intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
         no_intf_file="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
         start_year=1980, end_year=2010):
    """
    Study impact of interflow only for the cases when liquid soil moisture is greater than field capacity
    """

    img_folder = "intf_during_th_gt_thbfc"

    cache_file = "intf_during_th_gt_thbfc_{}_{}-{}.cache".format(ctypes.c_size_t(hash(intf_file + no_intf_file)).value,
                                                             start_year, end_year)


    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    with tb.open_file(intf_file) as intf_handle, tb.open_file(no_intf_file) as no_intf_handle:
        # Get runoff tables (level_index == 0 corresponds to runoff from soil)
        traf_table_intf = intf_handle.get_node("/TRAF")
        traf_table_no_intf = no_intf_handle.get_node("/TRAF")

        th_table_intf = intf_handle.get_node("/I0")
        th_table_no_intf = no_intf_handle.get_node("/I0")

        bfc = get_bulk_field_capacity()

        if os.path.isfile(cache_file):
            total_diff = pickle.load(open(cache_file))
        else:
            total_diff = get_runoff_differences_composit(traf_table_intf=traf_table_intf, th_table_intf=th_table_intf,
                                                         traf_table_no_intf=traf_table_no_intf,
                                                         th_table_no_intf=th_table_no_intf, thbfc_field=bfc,
                                                         start_year=start_year, end_year=end_year)

            pickle.dump(total_diff, open(cache_file, "wb"))

        # save the figure
        plt.figure()
        total_diff /= float(end_year - start_year + 1)

        # Convert to mm
        total_diff *= 3 * 60 * 60
        lons, lats, bm = analysis.get_basemap_from_hdf(intf_file)

        x, y = bm(lons, lats)

        clevs = [0.5, 1, 5, 30, 100, 150]
        clevs = [-c for c in reversed(clevs)] + clevs

        cmap = cm.get_cmap("bwr", len(clevs) - 1)
        bn = BoundaryNorm(clevs, len(clevs) - 1)
        im = bm.pcolormesh(x, y, total_diff, cmap=cmap, norm=bn)
        bm.drawcoastlines()
        bm.colorbar(im, ticks=clevs)
        plt.savefig(os.path.join(img_folder, "traf_diff_{}-{}.png".format(start_year, end_year)))



    pass


if __name__ == '__main__':
    main()