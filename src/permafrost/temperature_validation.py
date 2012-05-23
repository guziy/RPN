from netCDF4 import Dataset
import os
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import maskoceans, Basemap
import application_properties
import my_colormaps
from active_layer_thickness import CRCMDataManager
from rpn import level_kinds
from rpn.rpn import RPN
from util import plot_utils

__author__ = 'huziy'

import numpy as np
from cru.temperature import CRUDataManager
import matplotlib.pyplot as plt
import draw_regions
import matplotlib as mpl


def compare_all_seasons():
    start_year = 1958
    end_year = 1961

#    month_cols = [range(3,6), range(6, 9), range(9,12), [12,1,2]]
#    period_abs = ["mam", "jja", "son", "djf"]
    month_cols = [[12,1,2]]
    period_abs = ["djf"]
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
    #b, lons2d, lats2d = draw_regions.get_basemap_and_coords(llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10)
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()
    lons2d[lons2d > 180] -= 360
    x, y = b(lons2d, lats2d)


    cru = CRUDataManager()
    cru_data = cru.get_mean(start_year,end_year, months = the_months)
    cru_data_interp = cru.interpolate_data_to(cru_data, lons2d, lats2d)


    temp_levels = np.arange(-40, 40, 5)
    diff_levels = np.arange(-10, 12, 2)
    gs = gridspec.GridSpec(3,2)
    #plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=20, font_size=12)
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
    era40_folder = "data/CORDEX/na/era40_1"
    file_prefix = "dm"
    for file_name in os.listdir(era40_folder):
        if period_str.upper() in file_name and file_name.startswith(file_prefix):
            file_path = os.path.join(era40_folder, file_name)
            break
    #get the temperature
    rpn_obj = RPN(file_path)
    t2m_era40 = rpn_obj.get_first_record_for_name_and_level(varname="TT",
            level=1, level_kind=level_kinds.HYBRID)
    t2m_era40 = maskoceans(lons2d, lats2d, t2m_era40)
    ax = fig.add_subplot(gs[1,0])
    axes_list.append(ax)
    img = b.contourf(x, y, t2m_era40, ax = ax, levels = temp_levels)
    ax.set_title("ERA40 driven 1 (1958-1961)")
    plot_utils.draw_colorbar(fig, img, ax = ax)
    rpn_obj.close()

    #era40 - cru
    ax = fig.add_subplot(gs[1,1])
    axes_list.append(ax)
    img = b.contourf(x, y, t2m_era40 - cru_data_interp, ax = ax, levels = diff_levels)
    ax.set_title("ERA40 driven 1 - CRU")
    plot_utils.draw_colorbar(fig, img, ax = ax)


    plot_e2_data = False
    if plot_e2_data:
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





    fig = plt.figure()
    ax = plt.gca()
    img = b.contourf(x, y, t2m_era40 - cru_data_interp, ax = ax, levels = diff_levels)
    ax.set_title("ERA40 driven 1 - CRU")
    plot_utils.draw_colorbar(fig, img, ax = ax)
    b.drawcoastlines(ax = ax, linewidth = coast_line_width)
    b.contour(x, y, pf_kinds, ax = ax, colors = "k")
    fig.savefig("temperature_diff_{0}.png".format(period_str))

    pass

def _get_thawing_index(daily_temp):
    """
    daily temp float air(time, level, rlat, rlon) ;
     air:units = "Celsius" ;
    """


def validate_thawing_index():
    start_year = 1980
    end_year = 1996

    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"

    sim_names = ["ERA40","MPI","CanESM"]
    fold = "/home/samira/skynet/DailyTempData"
    simname_to_path = {
        "ERA40": os.path.join(fold, "TempERA40_b1_1981-2008.nc"),
        "MPI": os.path.join(fold, "TempMPI1981-2010.nc"),
        "CanESM":os.path.join(fold, "TempCanESM1981-2010.nc"),
    }


    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74
    )
    assert isinstance(basemap, Basemap)

    lons2d[lons2d > 180] -= 360

    om = CRUDataManager()
    clim = om.get_daily_climatology(start_year, end_year)
    obs = om.get_thawing_index_from_climatology(clim)
    obs = om.interpolate_data_to(obs, lons2d, lats2d, nneighbours=1) #interpolatee to the model grid




    x, y = basemap(lons2d, lats2d)
    #x = (x[1:,1:] + x[:-1, :-1]) /2.0


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3)

    #plot_utils.apply_plot_params(width_pt=None, width_cm=35,height_cm=55, font_size=35)
    fig = plt.figure()
    assert isinstance(fig, Figure)


    cmap = my_colormaps.get_red_blue_colormap(ncolors=10, reversed=True)
    gs = gridspec.GridSpec(3,1)

    all_axes = []
    all_img = []


    i = 0
    for name in sim_names:
        path = simname_to_path[name]

        ds = Dataset(path)
        data_mod = ds.variables["air"][:]
        mod = _get_thawing_index()



        delta = mod - obs
        ax = fig.add_subplot(gs[i,0])
        assert isinstance(ax, Axes)
        delta = np.ma.masked_where(mask_cond, delta)
        img = basemap.pcolormesh(x, y, delta, cmap = cmap, vmin=None, vmax = None)
        if not i:
            ax.set_title("Thawing index, Mod - Obs, ({0} - {1}) \n".format(start_year, end_year))
        i += 1
        ax.set_ylabel(name)
        all_axes.append(ax)
        all_img.append(img)



    i = 0
    axs_to_hide = []
    #zones and coastlines
    for the_ax, the_img in zip(all_axes, all_img):
        divider = make_axes_locatable(the_ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = fig.colorbar(the_img,  cax = cax, extend = "both")
        cax.set_title("$^{\\circ} {\\rm C}$\n")
        #cax.set_title("%\n")
        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=1.5)
        basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
                ax=the_ax, linewidth=3)

        if i != 1:
            axs_to_hide.append(cax)
        i += 1

    fig.tight_layout()

    for the_ax in axs_to_hide:
        the_ax.set_visible(False)

    fig.savefig("tmp_validation_era_mpi_canesm.png")

    pass


def validate_using_monthly_diagnostics():
    start_year = 1980
    end_year = 1996




    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"

    sim_names = ["ERA40","MPI","CanESM"]
    simname_to_path = {
        "ERA40": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1_dm",
        "MPI": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_MPI_B1_dm",
        "CanESM": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1_dm"
    }


    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=45.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74,
        anchor = "W"
    )
    assert isinstance(basemap, Basemap)

    lons2d[lons2d > 180] -= 360

    obs_manager = CRUDataManager()
    obs = obs_manager.get_mean(start_year, end_year, months=[6,7,8])
    obs = obs_manager.interpolate_data_to(obs, lons2d, lats2d, nneighbours=1)



    x, y = basemap(lons2d, lats2d)

    #x = (x[1:,1:] + x[:-1, :-1]) /2.0


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 2)

    #plot_utils.apply_plot_params(width_pt=None, width_cm=35,height_cm=55, font_size=35)
    fig = plt.figure()
    assert isinstance(fig, Figure)


    cmap = my_colormaps.get_red_blue_colormap(ncolors=10, reversed=True)
    gs = gridspec.GridSpec(3,2, width_ratios=[1,0.1], hspace=0, wspace=0,
        left=0.05, bottom = 0.01, top=0.95)

    all_axes = []
    all_img = []


    i = 0
    for name in sim_names:
        path = simname_to_path[name]
        dm = CRCMDataManager(data_folder=path)
        mod = dm.get_mean_over_months_of_2d_var(start_year, end_year, months = [6,7,8],
            var_name="TT", level=1, level_kind=level_kinds.HYBRID)

        delta = mod - obs
        ax = fig.add_subplot(gs[i,0])
        assert isinstance(ax, Axes)
        delta = np.ma.masked_where(mask_cond, delta)
        img = basemap.pcolormesh(x, y, delta, cmap = cmap, vmin=-5.0, vmax = 5.0)
        if not i:
            ax.set_title("T2m, Mod - Obs, ({0} - {1}), JJA \n".format(start_year, end_year))
        i += 1
        #ax.set_ylabel(name)
        all_axes.append(ax)
        all_img.append(img)



    i = 0
    axs_to_hide = []
    #zones and coastlines
    for the_ax, the_img in zip(all_axes, all_img):
#        divider = make_axes_locatable(the_ax)
#        cax = divider.append_axes("right", "5%", pad="3%")
        #cax.set_title("%\n")
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
    cax.set_anchor("W")
    cax.set_aspect(30)
    formatter = FuncFormatter(
        lambda x, pos: "{0: <6}".format(str(x))
    )
    cb = fig.colorbar(all_img[0], ax = cax, cax = cax,
        extend = "both", format = formatter)
    cax.set_title("$^{\\circ} {\\rm C}$")

    #fig.tight_layout(h_pad=0)



#    for the_ax in axs_to_hide:
#        the_ax.set_visible(False)

    fig.savefig("tmp_validation_era_mpi_canesm.png")





if __name__ == "__main__":
    plot_utils.apply_plot_params(width_pt=None, width_cm=28, height_cm=40, font_size=25)
    application_properties.set_current_directory()
    #compare_all_seasons()
    validate_using_monthly_diagnostics()
    print "Hello world"
  