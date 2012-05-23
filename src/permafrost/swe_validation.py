import os
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Affine2D, Bbox
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import maskoceans, Basemap, cm
from active_layer_thickness import CRCMDataManager
import matplotlib.pyplot as plt
import application_properties
import my_colormaps
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
    #plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=30, font_size=12)
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



def validate_using_monthly_diagnostics():
    start_year = 1980
    end_year = 1996




    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"

    sim_names = ["ERA40","MPI","CanESM"]
    simname_to_path = {
        "ERA40": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1",
        "MPI": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_MPI_B1",
        "CanESM": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1"
    }


    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=45.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74,
        anchor="W"
    )
    assert isinstance(basemap, Basemap)

    lons2d[lons2d > 180] -= 360

    swe_obs_manager = SweDataManager(var_name="SWE")
    swe_obs = swe_obs_manager.get_mean(start_year, end_year, months=[12,1,2])
    swe_obs = swe_obs_manager.interpolate_data_to(swe_obs, lons2d, lats2d, nneighbours=1)



    x, y = basemap(lons2d, lats2d)
    #x = (x[1:,1:] + x[:-1, :-1]) /2.0


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 2)

    #plot_utils.apply_plot_params(width_pt=None, width_cm=35,height_cm=55, font_size=35)
    fig = plt.figure()
    assert isinstance(fig, Figure)


    cmap = my_colormaps.get_red_blue_colormap(ncolors=10)
    gs = gridspec.GridSpec(3,2, width_ratios=[1,0.1], hspace=0, wspace=0,
        left=0.05, bottom = 0.01, top=0.95)


    all_axes = []
    all_img = []


    i = 0
    for name in sim_names:
        path = simname_to_path[name]
        dm = CRCMDataManager(data_folder=path)
        swe_mod = dm.get_mean_over_months_of_2d_var(start_year, end_year, months = [12,1,2], var_name="I5")

        delta = swe_mod - swe_obs
        ax = fig.add_subplot(gs[i,0])
        assert isinstance(ax, Axes)
        delta = np.ma.masked_where(mask_cond, delta)
        img = basemap.pcolormesh(x, y, delta, cmap = cmap, vmin=-100, vmax = 100)
        if not i:
            ax.set_title("SWE, Mod - Obs, ({0} - {1}), DJF\n".format(start_year, end_year))
        i += 1
        #ax.set_ylabel(name)
        all_axes.append(ax)
        all_img.append(img)



    i = 0
    axs_to_hide = []
    #zones and coastlines
    for the_ax, the_img in zip(all_axes, all_img):
        #divider = make_axes_locatable(the_ax)
        #cax = divider.append_axes("bottom", "5%", pad="3%")

        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=0.5)
        basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
                ax=the_ax, linewidth=1.5, drawbounds=False)

        for nshape,seg in enumerate(basemap.zone):
            if basemap.zone_info[nshape]["EXTENT"] != "C": continue
            poly = mpl.patches.Polygon(seg,edgecolor = "k", facecolor="none", zorder = 10, lw = 1.5)
            the_ax.add_patch(poly)

        i += 1

    cax = fig.add_subplot(gs[:,1])
    assert isinstance(cax, Axes)
    cax.set_anchor("W")
    cax.set_aspect(30)

    formatter = FuncFormatter(
        lambda x, pos: "{0: <6}".format(str(x))
    )
    cb = fig.colorbar(all_img[0], ax = cax, cax = cax, extend = "both", format = formatter)

    cax.set_title("mm")
    print "aspect = ", cax.get_aspect()

    #fig.tight_layout(h_pad=0)

#    for the_ax in axs_to_hide:
#        the_ax.set_visible(False)

    fig.savefig("swe_validation_era_mpi_canesm_djf.png")





    #swe_obs = swe_obs_manager.get_mean(start_year, end_year, months=the_months)

    pass


if __name__ == "__main__":
    plot_utils.apply_plot_params(width_pt=None, width_cm=28, height_cm=40, font_size=25)
    application_properties.set_current_directory()
    #main()
    #compare_swe_diff_for_era40_driven()
    validate_using_monthly_diagnostics()
    print "Hello world"
  