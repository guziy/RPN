import os
from matplotlib import gridspec
from mpl_toolkits.basemap import maskoceans
import application_properties
import level_kinds
from rpn import RPN
from util import plot_utils

__author__ = 'huziy'

import numpy as np
from cru.temperature import CRUDataManager
import matplotlib.pyplot as plt
import draw_regions


def compare_all_seasons():
    start_year = 1958
    end_year = 1974

    month_cols = [range(3,6), range(6, 9), range(9,12), [12,1,2]]
    period_abs = ["mam", "jja", "son", "djf"]

    for months, period_s in zip(month_cols, period_abs):
        compare_for_season(start_year = start_year, end_year=end_year,
           the_months=months, period_str=period_s
        )


def compare_for_season(   start_year = 1958,
    end_year = 1974,
    the_months = None,
    period_str = "djf"):

    """
    Compare CRU, ERA40-driven and GCM-driven s
    """
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10)
    lons2d[lons2d > 180] -= 360
    x, y = b(lons2d, lats2d)

    cru = CRUDataManager()
    cru_data = cru.get_mean(start_year,end_year, months = the_months)
    cru_data_interp = cru.interpolate_data_to(cru_data, lons2d, lats2d)


    temp_levels = np.arange(-40, 40, 5)
    diff_levels = np.arange(-10, 12, 2)
    gs = gridspec.GridSpec(3,2)
    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=20, font_size=12)
    fig = plt.figure()
    coast_line_width = 0.25
    axes_list = []



    #plot CRU data
    ax = fig.add_subplot(gs[0,:])
    axes_list.append(ax)
    cru_data_interp = maskoceans(lons2d, lats2d, cru_data_interp)
    img = b.contourf(x, y, cru_data_interp, ax = ax, levels = temp_levels)
    ax.set_title("CRU")
    plot_utils.draw_colorbar(fig, img, ax = ax)


    #era40 driven
    file_path = None
    era40_folder = "data/CORDEX/na/means_month/era40_driven"
    for file_name in os.listdir(era40_folder):
        if period_str.upper() in file_name:
            file_path = os.path.join(era40_folder, file_name)
            break
    #get the temperature
    rpn_obj = RPN(file_path)
    t2m = rpn_obj.get_first_record_for_name_and_level(varname="TT",
            level=1, level_kind=level_kinds.HYBRID)
    t2m = maskoceans(lons2d, lats2d, t2m)
    ax = fig.add_subplot(gs[1,0])
    axes_list.append(ax)
    img = b.contourf(x, y, t2m, ax = ax, levels = temp_levels)
    ax.set_title("ERA40 driven")
    plot_utils.draw_colorbar(fig, img, ax = ax)

    #era40 - cru
    ax = fig.add_subplot(gs[1,1])
    axes_list.append(ax)
    img = b.contourf(x, y, t2m - cru_data_interp, ax = ax, levels = diff_levels)
    ax.set_title("ERA40 driven - CRU")
    plot_utils.draw_colorbar(fig, img, ax = ax)



    ##get and plot E2 data
    file_path = None
    e2_folder = "data/CORDEX/na/e2"
    prefix = "dm"
    #get file path
    for file_name in os.listdir(e2_folder):
        if file_name.endswith(period_str) and file_name.startswith(prefix):
            file_path = os.path.join(e2_folder, file_name)
            break
        pass
    #get the temperature
    rpn_obj = RPN(file_path)
    t2m = rpn_obj.get_first_record_for_name_and_level(varname="TT",
            level=1, level_kind=level_kinds.HYBRID)
    t2m = maskoceans(lons2d, lats2d, t2m)
    ax = fig.add_subplot(gs[2,0])
    axes_list.append(ax)
    img = b.contourf(x, y, t2m, ax = ax, levels = temp_levels)
    ax.set_title("E2, GCM driven")
    plot_utils.draw_colorbar(fig, img, ax = ax)

    #e2 - cru
    ax = fig.add_subplot(gs[2,1])
    axes_list.append(ax)
    img = b.contourf(x, y, t2m - cru_data_interp, ax = ax, levels = diff_levels)
    ax.set_title("E2, GCM driven - CRU")
    plot_utils.draw_colorbar(fig, img, ax = ax)


    ####Draw common elements
    pf_kinds = draw_regions.get_permafrost_mask(lons2d, lats2d)
    for the_ax in axes_list:
        b.drawcoastlines(ax = the_ax, linewidth = coast_line_width)
        b.contour(x, y, pf_kinds, ax = the_ax, colors = "k")



    gs.tight_layout(fig, h_pad = 5, w_pad = 5, pad=2)
    fig.suptitle(period_str.upper(), y = 0.03, x = 0.5)
    fig.savefig("temperature_validation_{0}.png".format(period_str))
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    compare_all_seasons()
    print "Hello world"
  