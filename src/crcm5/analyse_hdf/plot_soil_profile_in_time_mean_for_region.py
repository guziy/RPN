from datetime import datetime
from math import floor
import os
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.dates import date2num, DateFormatter, MonthLocator
from matplotlib.ticker import MaxNLocator
from crcm5 import infovar
from util.geo.index_shapes import IndexRectangle, IndexPoint

__author__ = 'huziy'

import matplotlib.pyplot as plt
from . import common_plot_params as cpp
import numpy as np
from class_scheme import configuration as class_conf
from matplotlib import cm
from .input_params import InputParams


def main_compare_two_simulations():
    """
    draw (params2 - params0) climatologic profiles, and a region of interest
    """
    #folder for storing result images
    img_folder = ""
    start_date = datetime(1980, 1, 1)
    end_date = datetime(2010, 12, 31)


    path0 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    path2 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"

    exp_label = "interflow_effect_soil"

    img_folder = "images_for_lake-river_paper"
    img_folder = os.path.join(img_folder, exp_label)
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)


    rectangle = IndexRectangle(
        lower_left_point=IndexPoint(40, 20),
        width=100, height=45
    )

    params2 = InputParams(hdf_path = path2,
                          is_for_comparison=True, start_date=start_date, end_date=end_date, rectangle=rectangle)

    params0 = InputParams(hdf_path=path0,
                          is_for_comparison=True, start_date=start_date, end_date=end_date, rectangle=rectangle)

    imin, jmin, w, h = params0.get_start_end_indices_of_selected_region()

    i_sel, j_sel = np.where(params0.get_land_mask_using_flow_dirs())

    i_sel_1 = i_sel[(i_sel >= imin) & (i_sel < imin + w) & (j_sel >= jmin) & (j_sel < jmin + h)]
    j_sel_1 = j_sel[(i_sel >= imin) & (i_sel < imin + w) & (j_sel >= jmin) & (j_sel < jmin + h)]

    i_sel = i_sel_1
    j_sel = j_sel_1

    levs2d, dnums2d = None, None

    #plot the profile
    fig = plt.figure()
    gs = gridspec.GridSpec(len(params0.var_list), 2)

    #The number of levels of interest
    n_select_level = 5

    #calculate and plot differences
    for vindex, var_name in enumerate(params0.var_list):
        print("plotting {0} ...".format(var_name))
        dates, levels, data2 = params2.calculate_mean_clim_for_3d_var(var_name=var_name)
        _, _, data0 = params0.calculate_mean_clim_for_3d_var(var_name=var_name)

        data = data2 - data0

        #calculate the profile
        selected_diff = data[:, :n_select_level, i_sel, j_sel]
        sel_data = (data0[:, :n_select_level, i_sel, j_sel] + data2[:, :n_select_level, i_sel, j_sel]) * 0.5
        selected_mean = np.zeros_like(sel_data)
        where_to_compute = np.abs(selected_diff) > 0
        selected_mean[where_to_compute] = selected_diff[where_to_compute] / sel_data[where_to_compute] * 100.0

        selected_mean = selected_mean.mean(axis=2)



        #rectangle subplot
        ax = fig.add_subplot(gs[:, 0])
        params0.basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH, ax=ax)
        ax.add_patch(params0.get_mpl_rectangle_for_selected_region())

        #profile subplot
        ax = fig.add_subplot(gs[vindex, 1])
        assert isinstance(ax, Axes)


        if levs2d is None:
            ax.set_ylabel("Depth (m)")
            levels_meters = np.cumsum([0, ] + class_conf.level_width_list_26_default)[:-1][:n_select_level]
            dnums = date2num(dates)
            levs2d, dnums2d = np.meshgrid(levels_meters, dnums)




        vmin, vmax = selected_mean.min(), selected_mean.max()
        d = max(abs(vmin), abs(vmax))
        ncolors = 11
        cmap = cm.get_cmap("RdBu_r", ncolors)
        color_levs = np.linspace(-d, d, ncolors + 1)

        step = color_levs[1] - color_levs[0]
        ndec = abs(floor(np.log10(step)))
        color_levs = np.round(color_levs, decimals=int(ndec))

        img = ax.contourf(dnums2d, levs2d, selected_mean, cmap = cmap, levels = color_levs)
        cb = plt.colorbar(img, ticks = color_levs[::2])
        cb.ax.set_aspect(10)

        ax.xaxis.set_major_formatter(DateFormatter("%d\n%b"))
        ax.xaxis.set_major_locator(MonthLocator(bymonth=list(range(1, 13, 2))))
        if vindex < len(params0.var_list) - 1:
            ax.xaxis.set_ticklabels([])
        ax.invert_yaxis()
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        #ax.grid(b = True)
        ax.annotate(infovar.get_display_label_for_var(var_name),
                    xy = (0.8, 0.2), xycoords = "axes fraction",
                    bbox = dict(facecolor = "w"))


    #construct the path to the output figure
    impath = os.path.join(img_folder, params0.get_imfilename_for_var(var_name = "_".join(params0.var_list)))

    #save the figure
    fig.savefig(impath, dpi=cpp.FIG_SAVE_DPI, bbox_inches = "tight")
    plt.close(fig)


def main_plot_for_one_simulation(input_params=None):
    """
    Plot an average profile of soil moisture or temperature or ice
    input_params: object containing input parameters. Uses climatological mean.

    """
    assert isinstance(input_params, InputParams)

    imin, jmin, w, h = input_params.get_start_end_indices_of_selected_region()

    #Read the 3D climatological field and levels, generate dates for a generic year (daily)
    for var_name in input_params.var_list:
        print("plotting {0} ...".format(var_name))

        #construct the path to the output figure
        impath = os.path.join(input_params.img_folder, input_params.get_imfilename_for_var(var_name=var_name))
        dates, levels, data = input_params.calculate_mean_clim_for_3d_var(var_name=var_name)

        levels = np.cumsum([0, ] + class_conf.level_width_list_26_default)[:-1]


        #calculate the profile
        selected_mean = data[:, :, imin:imin + w, jmin:jmin + h].mean(axis=2).mean(axis=2)


        #plot the profile
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2)

        #rectangle subplot
        ax = fig.add_subplot(gs[0, 0])
        input_params.basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH, ax=ax)
        ax.add_patch(input_params.get_mpl_rectangle_for_selected_region())

        #profile subplot
        ax = fig.add_subplot(gs[0, 1])
        assert isinstance(ax, Axes)
        dnums = date2num(dates)
        levs2d, dnums2d = np.meshgrid(levels, dnums)
        img = ax.contourf(dnums2d, levs2d, selected_mean)
        ax.invert_yaxis()

        ax.xaxis.set_major_formatter(DateFormatter("%d\n%b"))
        ax.xaxis.set_major_locator(MonthLocator(bymonth=list(range(1, 13, 2))))

        plt.colorbar(img)
        #save the figure
        fig.savefig(impath, dpi=cpp.FIG_SAVE_DPI)
        plt.close(fig)




def exp_plot_one_simulation():
    main_plot_for_one_simulation(input_params=InputParams())


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    #exp_plot_one_simulation()