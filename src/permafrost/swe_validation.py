import os
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import maskoceans
from active_layer_thickness import CRCMDataManager
import matplotlib.pyplot as plt
import application_properties
from util import plot_utils

__author__ = 'huziy'

import numpy as np
import matplotlib as mpl
from matplotlib import colors
from swe import SweDataManager
import draw_regions

def draw_colorbar(fig, img, ax = None, boundaries = None, ticks = None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax, boundaries = boundaries, ticks = ticks)


def compare_swe_diff_for_era40_driven():
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10)

    lons2d[lons2d > 180] -= 360

    x, y = b(lons2d, lats2d)
    #period
    start_year = 1981
    end_year = 1997
    the_months = [12,1,2]

    levels = [10] + range(20, 120, 20) + [150,200, 300,500,1000]
    cmap = mpl.cm.get_cmap(name="jet_r", lut = len(levels))
    norm = colors.BoundaryNorm(levels, cmap.N)



    swe_obs_manager = SweDataManager(var_name="SWE")
    swe_obs = swe_obs_manager.get_mean(start_year, end_year, months=the_months)
    print "Calculated obs swe"

    swe_obs_interp = swe_obs_manager.interpolate_data_to(swe_obs, lons2d, lats2d)


    axes_list = []

    levels_diff = np.arange(-100, 110, 10)

    #plot model res. (ERA40 driven 1)

    paths = ["data/CORDEX/na/era40_1", "data/CORDEX/na/era40_2"]
    prefixes = ["pmNorthAmerica_0.44deg_ERA40-Int_{0}_1958-1961".format("DJF"),
                "pmNorthAmerica_0.44deg_ERA40-Int2_{0}_1958-1961".format("DJF")
                ]
    pf_kinds = draw_regions.get_permafrost_mask(lons2d, lats2d)
    for i, the_path in enumerate(paths):
        base = os.path.basename(the_path)
        fig = plt.figure()
        ax = plt.gca()
        axes_list.append(ax)


        swe_model_era = CRCMDataManager.get_mean_2d_from_climatologies(path=the_path,
            var_name="I5", file_prefixes=prefixes)
        swe_model_era = maskoceans(lons2d, lats2d, swe_model_era)

        #plot model(ERA40 driven) - obs
        axes_list.append(ax)
        img = b.contourf(x, y, swe_model_era - swe_obs_interp, levels = levels_diff)
        draw_colorbar(fig, img, ax = ax, boundaries=levels_diff)
        ax.set_title("Model ({0} 1958-1961) - Obs.".format(base))

        b.drawcoastlines(ax = ax, linewidth = 0.2)
        b.contour(x, y, pf_kinds, ax = ax, colors = "k")
        fig.savefig("swe_{0}.png".format(base))





def main():

    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10)

    lons2d[lons2d > 180] -= 360

    x, y = b(lons2d, lats2d)
    #period
    start_year = 1981
    end_year = 1997
    the_months = [12,1,2]

    levels = [10] + range(20, 120, 20) + [150,200, 300,500,1000]
    cmap = mpl.cm.get_cmap(name="jet_r", lut = len(levels))
    norm = colors.BoundaryNorm(levels, cmap.N)



    swe_obs_manager = SweDataManager(var_name="SWE")
    swe_obs = swe_obs_manager.get_mean(start_year, end_year, months=the_months)
    print "Calculated obs swe"

    swe_obs_interp = swe_obs_manager.interpolate_data_to(swe_obs, lons2d, lats2d, nneighbours=1)

    gs = gridspec.GridSpec(2,3)
    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=30, font_size=12)
    fig = plt.figure()
    coast_line_width = 0.25
    axes_list = []

    #plot obs on its own grid but in the model's projection
    ax = fig.add_subplot(gs[0,0])
    axes_list.append(ax)
    x_obs, y_obs = b(swe_obs_manager.lons2d, swe_obs_manager.lats2d)
    swe_obs = maskoceans(swe_obs_manager.lons2d, swe_obs_manager.lats2d, swe_obs)
    img = b.contourf(x_obs, y_obs, swe_obs, levels = levels, norm = norm, cmap = cmap)
    draw_colorbar(fig, img, ax = ax)
    ax.set_title("Obs native grid")

    #plot obs interpolated
    ax = fig.add_subplot(gs[1,0])
    axes_list.append(ax)
    swe_obs_interp = maskoceans(lons2d, lats2d, swe_obs_interp)
    img = b.contourf(x, y, swe_obs_interp, levels = levels, norm = norm, cmap = cmap)
    draw_colorbar(fig, img, ax = ax)
    ax.set_title("Obs interpolated \n to model grid")



    #plot model res. (ERA40 driven)
    ax = fig.add_subplot(gs[0,1])
    axes_list.append(ax)

    prefixes = ["pmNorthAmerica_0.44deg_ERA40-Int_{0}_1958-1977".format(m) for m in ["Dec", "Jan", "Feb"]]
    swe_model_era = CRCMDataManager.get_mean_2d_from_climatologies(path="data/CORDEX/na/means_month/era40_driven",
        var_name="I5", file_prefixes=prefixes)
    swe_model_era = maskoceans(lons2d, lats2d, swe_model_era)
    img = b.contourf(x, y, swe_model_era, levels = levels, norm = norm, cmap = cmap)
    draw_colorbar(fig, img, ax = ax)
    ax.set_title("Model (ERA40 driven 1958-1977)")

    #plot model(ERA40 driven) - obs
    ax = fig.add_subplot(gs[0,2])
    axes_list.append(ax)
    levels_diff = np.arange(-100, 110, 10)
    img = b.contourf(x, y, swe_model_era - swe_obs_interp, levels = levels_diff)
    draw_colorbar(fig, img, ax = ax, boundaries=levels_diff)
    ax.set_title("Model (ERA40 driven 1958-1977) - Obs.")



    #plot model res. (GCM driven, E2)
    ax = fig.add_subplot(gs[1,1])
    axes_list.append(ax)
    path = "/skynet1_exec2/separovi/results/North_America/tests_E/all/means_season"
    prefix = "pmNorthAmerica_0.44deg_CanHistoE2_A1979-1997"
    suffixes = ["djf"]
    swe_model_gcm = CRCMDataManager.get_mean_2d_from_climatologies(path=path, file_prefixes=[prefix],
                    file_suffixes=suffixes, var_name="I5")
    swe_model_gcm = maskoceans(lons2d, lats2d, swe_model_gcm)
    print "model: min = {0}; max = {1}".format(np.ma.min(swe_model_gcm), np.ma.max(swe_model_gcm))
    img = b.contourf(x, y, swe_model_gcm, levels = levels, norm = norm, cmap = cmap)
    draw_colorbar(fig, img, ax = ax, boundaries=levels_diff)
    ax.set_title("Model (GCM driven, E2, 1979-1997)")




    #plot model(gcm driven) - obs
    ax = fig.add_subplot(gs[1,2])
    axes_list.append(ax)
    levels_diff = np.arange(-100, 110, 10)
    img = b.contourf(x, y, np.ma.array(swe_model_gcm) - swe_obs_interp, levels = levels_diff)
    draw_colorbar(fig, img, ax = ax)
    ax.set_title("Model (GCM driven) - Obs.")



    ####Draw common elements
    pf_kinds = draw_regions.get_permafrost_mask(lons2d, lats2d)
    for the_ax in axes_list:
        b.drawcoastlines(ax = the_ax, linewidth = coast_line_width)
        b.contour(x, y, pf_kinds, ax = the_ax, colors = "k")

    gs.tight_layout(fig, h_pad = 0.9, w_pad = 18)
    fig.savefig("swe_validation.png")

    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    #main()
    compare_swe_diff_for_era40_driven()
    print "Hello world"
  