import os
import pickle

import matplotlib
matplotlib.use("Agg")

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from crcm5 import infovar

__author__ = 'huziy'

import numpy as np
import tables as tb

import matplotlib.pyplot as plt
import crcm5.analyse_hdf.do_analysis_using_pytables as pt_analysis


def run_with_default_params():
    """
    Just for test runs with defaults

    """
    # intf_file = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"

    intf_file = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"
    no_intf_file = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    main(
        interflow_data_path=intf_file, base_data_path=no_intf_file,
        start_year=1980, end_year=1988
    )


def get_mean_diffs(interflow_data_path="", base_data_path="",
                   start_year=1980, end_year=2010, months_of_interest=(4, 5, 6, 7, 8, 9),
                   delete_cache=True):
    """
    Get mean differences for fixed variables, between interflow_data_path and base_data_path files
    :param interflow_data_path:
    :param base_data_path:
    :param start_year:
    :param end_year:
    :param months_of_interest:
    :return:
    """
    # Build the name of the cache file
    cache_file = "cache_extr_intf_effect{}-{}_{}.bin".format(start_year, end_year,
                                                             "-".join(str(m) for m in months_of_interest))

    # Do not use caching by default
    if delete_cache:
        os.remove(cache_file)

    if os.path.isfile(cache_file):
        return pickle.load(open(cache_file))

    precip_limit = 0.0  # at least it should rain
    tt_limit = 0  # and the oil should not be frozen

    traf_diff = None  # surface runoff difference
    prcip_diff = None
    drainage_diff = None  # drainage difference
    i1_diff = None  # soil moisture difference
    months_query = "{}".format("|".join(["(month=={})".format(m) for m in months_of_interest]))
    year_query = "(year >= {}) & (year <= {})".format(start_year, end_year)
    print "months_query = {}".format(months_query)

    depth_to_bedrock = pt_analysis.get_array_from_file(base_data_path, var_name=infovar.HDF_DEPTH_TO_BEDROCK_NAME)

    with tb.open_file(interflow_data_path) as h_intf:
        pr_intf_table = h_intf.get_node("/", "PR")
        tt_intf_table = h_intf.get_node("/", "TT")
        traf_intf_table = h_intf.get_node("/", "TRAF")
        tdra_intf_table = h_intf.get_node("/", "TDRA")
        i1_intf_table = h_intf.get_node("/", "I1")

        assert isinstance(pr_intf_table, tb.Table)
        assert isinstance(tt_intf_table, tb.Table)
        assert isinstance(traf_intf_table, tb.Table)
        assert isinstance(tdra_intf_table, tb.Table)

        print len(pr_intf_table), len(tt_intf_table), len(traf_intf_table)

        with tb.open_file(base_data_path) as h_nointf:

            pr_nointf_table = h_nointf.get_node("/", "PR")
            tt_nointf_table = h_nointf.get_node("/", "TT")
            traf_nointf_table = h_nointf.get_node("/", "TRAF")
            tdra_nointf_table = h_nointf.get_node("/", "TDRA")
            i1_nointf_table = h_nointf.get_node("/", "I1")

            assert isinstance(pr_nointf_table, tb.Table)
            assert isinstance(tt_nointf_table, tb.Table)
            assert isinstance(traf_nointf_table, tb.Table)
            assert isinstance(tdra_nointf_table, tb.Table)

            for rownum, pr_intf_row in enumerate(pr_intf_table.where("({}) & {}".format(months_query, year_query))):
                year, month, day, hour = [pr_intf_row[k] for k in ["year", "month", "day", "hour"]]
                # print year, month, day, hour

                pr_intf_field = pr_intf_row["field"]
                tt_intf_field = None
                traf_intf_field = None
                tdra_intf_field = None
                i1_intf_field = None

                pr_nointf_field = None
                tt_nointf_field = None
                traf_nointf_field = None
                tdra_nointf_field = None
                i1_nointf_field = None

                # Get air temperature and precipitation for the same time
                tt_query = "(year == {}) & (month == {}) & (day == {}) & (hour == {})".format(year, month, day, hour)
                traf_query = "{} & (level_index == {})".format(tt_query, 0)
                for tt_row in tt_intf_table.where(tt_query):
                    tt_intf_field = tt_row["field"]
                    break

                # print tt_intf_field.min(), tt_intf_field.max()


                for traf_row in traf_intf_table.where(traf_query):
                    traf_intf_field = traf_row["field"]
                    break

                for tdra_row in tdra_intf_table.where(traf_query):
                    tdra_intf_field = tdra_row["field"]
                    break

                for i1_row in i1_intf_table.where(traf_query):
                    i1_intf_field = i1_row["field"]
                    break


                # for no interflow simulation
                for tt_row in tt_nointf_table.where(tt_query):
                    tt_nointf_field = tt_row["field"]
                    break

                for pr_row in pr_nointf_table.where(tt_query):
                    pr_nointf_field = pr_row["field"]
                    break

                for traf_row in traf_nointf_table.where(traf_query):
                    traf_nointf_field = traf_row["field"]
                    break

                for tdra_row in tdra_nointf_table.where(traf_query):
                    tdra_nointf_field = tdra_row["field"]
                    break

                for i1_row in i1_nointf_table.where(traf_query):
                    i1_nointf_field = i1_row["field"]
                    break

                if traf_diff is None:
                    traf_diff = np.zeros(pr_intf_field.shape)
                    prcip_diff = np.zeros(pr_intf_field.shape)
                    drainage_diff = np.zeros(pr_intf_field.shape)
                    i1_diff = np.zeros(pr_intf_field.shape)

                points_of_interest = (
                    (pr_intf_field > precip_limit) & (pr_nointf_field > precip_limit) &
                    (tt_intf_field > tt_limit) & (tt_nointf_field > tt_limit)
                    & (abs(pr_intf_field - pr_nointf_field) < 0.01 * (pr_intf_field + pr_nointf_field) / 2.0)
                )

                if rownum % 100 == 0:
                    print "Precipitation ranges in M/s"
                    print pr_intf_field.min(), pr_intf_field.max()
                    print pr_nointf_field.min(), pr_nointf_field.max()

                if traf_intf_field is None:
                    print "intf field is none"
                    print traf_query

                if traf_nointf_field is None:
                    print "nointf field is none"
                    print traf_query

                traf_diff[points_of_interest] += traf_intf_field[points_of_interest] - \
                                                 traf_nointf_field[points_of_interest]

                prcip_diff[points_of_interest] += pr_intf_field[points_of_interest] - \
                                                  pr_nointf_field[points_of_interest]

                drainage_diff[points_of_interest] += tdra_intf_field[points_of_interest] - \
                                                     tdra_nointf_field[points_of_interest]

                i1_diff[points_of_interest] += i1_intf_field[points_of_interest] - \
                                               i1_nointf_field[points_of_interest]

                # if rownum % 100 == 0 and debug_plots:
                #     fig = plt.figure()
                #     im = plt.pcolormesh(traf_diff.transpose() * 3 * 60 * 60)
                #     plt.colorbar(im)
                #     plt.savefig("{}/{}.jpg".format(img_dir, rownum))
                #     plt.close(fig)
                #
                #     plt.figure()
                #     im = plt.pcolormesh(traf_intf_field.transpose() * 60 * 60 * 24)
                #     plt.colorbar(im)
                #     plt.savefig("{}/traf_{}.jpg".format(img_dir, rownum))
                #     plt.close(fig)

    pickle.dump([traf_diff, prcip_diff, drainage_diff, i1_diff], open(cache_file, "w"))
    return traf_diff, prcip_diff, drainage_diff, i1_diff


def main(interflow_data_path="", base_data_path="",
         start_year=1980, end_year=2010, months_of_interest=(4, 5, 6, 7, 8, 9),
         debug_plots=False, dt_seconds=3 * 60 * 60):
    # have a field diff accumulator: dacc
    # rain_regions1 = (PR1 > 0) & (I01[:,:,0] > 273.15)
    # rain_regions2 = (PR2 > 0) & (I02[:,:,0] > 273.15)
    # rain_regions = rain_regions1 & rain_regions2
    # dacc[rain_regions] += TRAF1[rain_regions]  <-- Runoff effect

    traf_diff, prcip_diff, tdra_diff, i1_diff = get_mean_diffs(interflow_data_path=interflow_data_path,
                                                               base_data_path=base_data_path,
                                                               start_year=start_year, end_year=end_year,
                                                               months_of_interest=months_of_interest)

    # Get coordinates of the fields
    lons, lats, bsm = pt_analysis.get_basemap_from_hdf(base_data_path)
    x, y = bsm(lons, lats)

    # Folder containing debug images
    img_dir = "test_intf_extr/exp_avoid_small_incrs_{}-{}".format(start_year, end_year)
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)


    # Plot surface runoff
    plt.figure()
    clevs = [5, 10, 20, 50]
    clevs = [-c for c in reversed(clevs)] + clevs
    cmap = cm.get_cmap(name="seismic", lut=len(clevs) - 1)
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    im = bsm.pcolormesh(x, y, traf_diff * dt_seconds / float(end_year - start_year + 1), norm=bn,
                        cmap=cmap)
    plt.colorbar(im)
    plt.title("Surface runoff (mm/year)")
    bsm.drawcoastlines(linewidth=0.5)
    plt.savefig("{}/traf_diff_traf_due_to_intf_{}_{}-{}.jpg".format(
        img_dir, "-".join(str(m) for m in months_of_interest), start_year, end_year))

    # Plot precipitation
    plt.figure()
    clevs = [1, 2, 5, 10, 20, 40]
    clevs = [-c for c in reversed(clevs)] + clevs
    cmap = cm.get_cmap(name="seismic", lut=len(clevs) - 1)
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    im = bsm.pcolormesh(x, y, prcip_diff * dt_seconds * 1000.0 / float(end_year - start_year + 1),
                        cmap=cmap, norm=bn)
    plt.colorbar(im)
    plt.title("Precipitation (mm/year)")
    bsm.drawcoastlines(linewidth=0.5)
    plt.savefig("{}/precip_diff_traf_due_to_intf_{}_{}-{}.jpg".format(
        img_dir, "-".join(str(m) for m in months_of_interest), start_year, end_year))

    ###Plot drainage
    plt.figure()
    clevs = [1, 2, 5, 10, 20, 40]
    clevs = [-c for c in reversed(clevs)] + clevs
    cmap = cm.get_cmap(name="seismic", lut=len(clevs) - 1)
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    im = bsm.pcolormesh(x, y, tdra_diff * dt_seconds / float(end_year - start_year + 1), cmap=cmap, norm=bn)
    plt.colorbar(im)
    plt.title("Drainage (mm/year)")
    bsm.drawcoastlines(linewidth=0.5)
    plt.savefig("{}/tdra_diff_traf_due_to_intf_{}_{}-{}.jpg".format(
        img_dir, "-".join(str(m) for m in months_of_interest), start_year, end_year))

    ###Plot first soil layer moisture
    plt.figure()
    clevs = [1, 2, 5, 10, 20, 40]
    clevs = [-c for c in reversed(clevs)] + clevs
    cmap = cm.get_cmap(name="seismic", lut=len(clevs) - 1)
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    im = bsm.pcolormesh(x, y, i1_diff / float(end_year - start_year + 1), cmap=None, norm=None)
    plt.colorbar(im)
    plt.title("Liquid soil moisture (mm)")
    bsm.drawcoastlines(linewidth=0.5)
    plt.savefig("{}/I1_diff_due_to_intf_{}_{}-{}.jpg".format(
        img_dir, "-".join(str(m) for m in months_of_interest), start_year, end_year))


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    run_with_default_params()
