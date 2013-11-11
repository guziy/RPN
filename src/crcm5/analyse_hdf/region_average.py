import os
import brewer2mpl
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import pandas

__author__ = 'huziy'

import do_analysis_using_pytables as analysis
import matplotlib.pyplot as plt
import common_plot_params as cpp
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
#for making time series from area averaged values


def main():
    # brewer2mpl.get_map args: set name  set type  number of colors
    bmap = brewer2mpl.get_map("Set1", "qualitative", 9)
    # Change the default colors
    mpl.rcParams["axes.color_cycle"] = bmap.mpl_colors

    varnames = ["STFL", "TRAF"]
    units = [r"${\rm ^{\circ} C}$", "mm/d"]
    factors = [60 * 60 * 24 * 1000, 60 * 60 * 24 * 1000]
    levels = [None, 1]
    start_year = 1979
    end_year = 1985

    images_folder = "images_for_lake-river_paper/area_averages"
    if not os.path.isdir(images_folder):
        os.mkdir(images_folder)

    path1 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf"
    label1 = "CRCM5-HCD-RL"

    path2 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf"
    label2 = "CRCM5-HCD-RL-INTFL"

    #region specification
    ll_crnr_i = 0
    ll_crnr_j = 0
    ur_crnr_i = ll_crnr_i + 200
    ur_crnr_j = ll_crnr_j + 200

    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=path1)
    x, y = basemap(lons2d, lats2d)

    fig = plt.figure()
    assert isinstance(fig, Figure)
    gs = gridspec.GridSpec(1 + len(varnames), 2)
    ax_map = fig.add_subplot(gs[0, :])
    basemap.drawcoastlines(ax = ax_map, linewidth=cpp.COASTLINE_WIDTH)

    xll = x[ll_crnr_i, ll_crnr_j]
    yll = y[ll_crnr_i, ll_crnr_j]

    xur = x[ur_crnr_i, ur_crnr_j]
    yur = y[ur_crnr_i, ur_crnr_j]

    assert isinstance(ax_map, Axes)
    ax_map.add_patch(
        Rectangle((xll, yll), xur - xll, yur - yll, facecolor = "green", alpha=0.5)
    )

    mask = (x <= xur) & (x >= xll) & (y <= yur) & (y >= yll)
    i_arr, j_arr = np.where(mask)

    for i, varname in enumerate(varnames):
        level = levels[i]

        ax_graph = fig.add_subplot(gs[1, i])
        ax_graph.set_title("{0}, {1}".format(varname, units[i]))
        t, v1 = analysis.get_daily_climatology(path_to_hdf_file=path1, var_name=varname, level=level,
                                               start_year=start_year, end_year=end_year)

        _, v2 = analysis.get_daily_climatology(path_to_hdf_file=path2, var_name=varname, level=level,
                                               start_year=start_year, end_year=end_year)

        v1 = np.mean(v1[:, i_arr, j_arr], axis=1)
        df1 = pandas.DataFrame(index = t, data = v1)
        df1 = pandas.rolling_mean(df1, window = 1, freq="10D")[:-1]
        t1 = df1.index
        v1 = df1.values * factors[i]

        v2 = np.mean(v2[:, i_arr, j_arr], axis=1)
        df2 = pandas.DataFrame(index = t, data = v2)
        df2 = pandas.rolling_mean(df2, window = 1, freq="10D")[:-1]
        v2 = df2.values * factors[i]

        dv = (v2 - v1)

        ax_graph.plot(t1, v1, label = label1, lw = 1)
        ax_graph.plot(t1, v2, label = label2, lw = 1)
        ax_graph.plot(t1, dv, label = "({0})-({1})".format(label2, label1), lw = 1)
        ax_graph.xaxis.set_major_formatter(DateFormatter("%b"))
        ax_graph.xaxis.set_minor_locator(MonthLocator())
        ax_graph.xaxis.set_major_locator(MonthLocator(bymonth=range(1, 13, 2)))
        ax_graph.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_graph.grid()

        if i == 0:
            ax_graph.legend(loc=(0.0, -0.7), borderaxespad=0.5, ncol=2)


    figpath = os.path.join(images_folder, "{0}_{1}_{2}_{3}_{4}_{5}.jpeg".format(
        ll_crnr_i, ll_crnr_j, ur_crnr_i, ur_crnr_j, "_vs_".join([label1, label2]), "-".join(varnames)
    ))

    fig.savefig(figpath, dpi = cpp.FIG_SAVE_DPI, bbox_inches = "tight")



if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()