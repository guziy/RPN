import glob
from collections import OrderedDict
from pathlib import Path

from matplotlib.colors import BoundaryNorm, from_levels_and_colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap

from application_properties import main_decorator
from crcm5.global_simulation import commons

from rpn.rpn_multi import MultiRPN
import numpy as np
import matplotlib.pyplot as plt

from util import plot_utils
from mpl_toolkits.basemap import cm as cm_basemap

LINEWIDTH = 0.3


def plot_seasonal_winds(level_hpa=850):
    pass


def plot_seasonal_precipitation(samples_folder, seasons=commons.default_seasons):
    vname = "PR"
    file_prefix = "pm"

    plot_units = "mm/day"
    mult_coeff = 1000 * 24 * 3600
    add_offset = 0

    out_dx = 0.5



    season_to_mean = OrderedDict()
    r = None
    for sname, months in seasons.items():

        # find files for the season
        paths = []
        for m in months:
            paths.extend(glob.glob(str(samples_folder.joinpath("*{:02d}/{}*".format(m, file_prefix)))))


        r = MultiRPN(paths)

        data = r.get_4d_field(varname=vname)
        season_to_mean[sname] = np.array([list(v.items())[0][1] for k, v in data.items()]).mean(axis=0)
        print("Processed: {}".format(sname))

    lons2d, lats2d = r.get_longitudes_and_latitudes_of_the_last_read_rec()



    img_folder = samples_folder.joinpath("images/seasonal")
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    img_file = img_folder.joinpath("{}.png".format(vname))

    plot_utils.apply_plot_params(width_cm=25, height_cm=15, font_size=18)
    fig = plt.figure()
    ncols = 2
    gs = GridSpec(len(season_to_mean) // ncols + int(not (len(season_to_mean) % ncols == 0)), ncols, wspace=0, hspace=0)
    xx, yy = None, None
    bmp = Basemap(projection="robin", lon_0=0)


    nlevs = 20
    clevs = None
    for i, (sname, field) in enumerate(season_to_mean.items()):
        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col])

        lons, lats, data_out = commons.interpolate_to_uniform_global_grid(field, lons_in=lons2d, lats_in=lats2d, out_dx=out_dx)

        if xx is None:
            xx, yy = bmp(lons, lats)


        cs = bmp.contourf(xx, yy, data_out * mult_coeff, nlevs if clevs is None else clevs, cmap=cm_basemap.s3pcpn_l, extend="max")

        # save color levels for next subplots
        clevs = cs.levels

        print(np.max(data_out * mult_coeff))

        ax.set_title(sname)
        cb = plt.colorbar(cs, ax=ax)
        if not (row == 0 and col == ncols - 1):
            # cb.ax.set_title(plot_units)
            cb.ax.set_visible(False)
        bmp.drawcoastlines(ax=ax, linewidth=LINEWIDTH)

    fig.suptitle("Precipitation, {}".format(plot_units))

    with img_file.open("wb") as f:
        fig.savefig(f, bbox_inches="tight")


def plot_seasonal_2m_temp(samples_folder, seasons=commons.default_seasons):
    vname = "TT"
    file_prefix = "dm"

    plot_units = "$^\circ$C"
    mult_coeff = 1
    add_offset = 0

    out_dx = 0.5
    level = 1  # in hybrid coords



    season_to_mean = OrderedDict()
    r = None
    for sname, months in seasons.items():

        # find files for the season
        paths = []
        for m in months:
            paths.extend(glob.glob(str(samples_folder.joinpath("*{:02d}/{}*".format(m, file_prefix)))))


        r = MultiRPN(paths)

        data = r.get_4d_field(varname=vname)
        season_to_mean[sname] = np.array([v[level] for k, v in data.items()]).mean(axis=0)
        print("Processed: {}".format(sname))

    lons2d, lats2d = r.get_longitudes_and_latitudes_of_the_last_read_rec()



    img_folder = samples_folder.joinpath("images/seasonal")
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    img_file = img_folder.joinpath("{}.png".format(vname))

    plot_utils.apply_plot_params(width_cm=25, height_cm=15, font_size=18)
    fig = plt.figure()
    ncols = 2
    gs = GridSpec(len(season_to_mean) // ncols + int(not (len(season_to_mean) % ncols == 0)), ncols, wspace=0, hspace=0)
    xx, yy = None, None
    bmp = Basemap(projection="robin", lon_0=0)



    cmap = plt.get_cmap('bwr')

    clevs = np.arange(-30, 32, 2)


    for i, (sname, field) in enumerate(season_to_mean.items()):
        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col])

        lons, lats, data_out = commons.interpolate_to_uniform_global_grid(field, lons_in=lons2d, lats_in=lats2d, out_dx=out_dx)

        if xx is None:
            xx, yy = bmp(lons, lats)


        cs = bmp.contourf(xx, yy, data_out * mult_coeff, clevs, cmap=cmap, extend="both")

        # save color levels for next subplots
        clevs = cs.levels

        print(np.max(data_out * mult_coeff))

        ax.set_title(sname)
        cb = plt.colorbar(cs, ax=ax)
        if not (row == 0 and col == ncols - 1):
            # cb.ax.set_title(plot_units)
            cb.ax.set_visible(False)
        bmp.drawcoastlines(ax=ax, linewidth=LINEWIDTH)

    fig.suptitle("2m temperature, {}".format(plot_units))

    with img_file.open("wb") as f:
        fig.savefig(f, bbox_inches="tight")

    plt.close(fig)


def plot_seasonal_windspeed(samples_folder, seasons=commons.default_seasons):
    vname = "UV"
    long_name = "Wind speed"
    file_prefix = "dm"

    plot_units = "m/s"
    mult_coeff = 0.5144444444
    add_offset = 0

    out_dx = 0.5
    level = 1  # in hybrid coords



    season_to_mean = OrderedDict()
    r = None
    for sname, months in seasons.items():

        # find files for the season
        paths = []
        for m in months:
            paths.extend(glob.glob(str(samples_folder.joinpath("*{:02d}/{}*".format(m, file_prefix)))))


        r = MultiRPN(paths)

        data = r.get_4d_field(varname=vname)
        season_to_mean[sname] = np.array([v[level] for k, v in data.items()]).mean(axis=0)
        print("Processed: {}".format(sname))

    lons2d, lats2d = r.get_longitudes_and_latitudes_of_the_last_read_rec()



    img_folder = samples_folder.joinpath("images/seasonal")
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    img_file = img_folder.joinpath("{}.png".format(vname))

    plot_utils.apply_plot_params(width_cm=25, height_cm=15, font_size=18)
    fig = plt.figure()
    ncols = 2
    gs = GridSpec(len(season_to_mean) // ncols + int(not (len(season_to_mean) % ncols == 0)), ncols, wspace=0, hspace=0)
    xx, yy = None, None
    bmp = Basemap(projection="robin", lon_0=0)



    cmap = cm_basemap.GMT_no_green

    clevs = None


    for i, (sname, field) in enumerate(season_to_mean.items()):
        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col])

        lons, lats, data_out = commons.interpolate_to_uniform_global_grid(field, lons_in=lons2d, lats_in=lats2d, out_dx=out_dx)

        if xx is None:
            xx, yy = bmp(lons, lats)


        cs = bmp.contourf(xx, yy, data_out * mult_coeff, 20 if clevs is None else clevs, cmap=cmap, extend="max")

        # save color levels for next subplots
        clevs = cs.levels

        print(np.max(data_out * mult_coeff))

        ax.set_title(sname)
        cb = plt.colorbar(cs, ax=ax)
        if not (row == 0 and col == ncols - 1):
            # cb.ax.set_title(plot_units)
            cb.ax.set_visible(False)
        bmp.drawcoastlines(ax=ax, linewidth=LINEWIDTH)

    fig.suptitle("{}, {}".format(long_name, plot_units))

    with img_file.open("wb") as f:
        fig.savefig(f, bbox_inches="tight")

    plt.close(fig)


def plot_seasonal_vertical_velocity(samples_folder, seasons=commons.default_seasons):
    vname = "WW"
    long_name = "Vertical compoent of wind velocity (500mb)"
    file_prefix = "dp"

    plot_units = "m/s"
    mult_coeff = 0.5144444444
    add_offset = 0

    out_dx = 0.5
    level = 500  # millibars



    season_to_mean = OrderedDict()
    r = None
    for sname, months in seasons.items():

        # find files for the season
        paths = []
        for m in months:
            paths.extend(glob.glob(str(samples_folder.joinpath("*{:02d}/{}*".format(m, file_prefix)))))


        r = MultiRPN(paths)

        data = r.get_4d_field(varname=vname)
        season_to_mean[sname] = np.array([v[level] for k, v in data.items()]).mean(axis=0)
        print("Processed: {}".format(sname))

    lons2d, lats2d = r.get_longitudes_and_latitudes_of_the_last_read_rec()



    img_folder = samples_folder.joinpath("images/seasonal")
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    img_file = img_folder.joinpath("{}.png".format(vname))

    plot_utils.apply_plot_params(width_cm=25, height_cm=15, font_size=18)
    fig = plt.figure()
    ncols = 2
    gs = GridSpec(len(season_to_mean) // ncols + int(not (len(season_to_mean) % ncols == 0)), ncols, wspace=0, hspace=0)
    xx, yy = None, None
    bmp = Basemap(projection="robin", lon_0=0)



    cmap = cm_basemap.GMT_polar

    clevs = np.arange(-0.3, 0.31, 0.01)


    for i, (sname, field) in enumerate(season_to_mean.items()):
        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col])

        lons, lats, data_out = commons.interpolate_to_uniform_global_grid(field, lons_in=lons2d, lats_in=lats2d, out_dx=out_dx)

        if xx is None:
            xx, yy = bmp(lons, lats)


        cs = bmp.contourf(xx, yy, data_out * mult_coeff, 20 if clevs is None else clevs, cmap=cmap, extend="both")

        # save color levels for next subplots
        clevs = cs.levels

        print(np.max(data_out * mult_coeff))

        ax.set_title(sname)
        cb = plt.colorbar(cs, ax=ax)
        if not (row == 0 and col == ncols - 1):
            # cb.ax.set_title(plot_units)
            cb.ax.set_visible(False)
        bmp.drawcoastlines(ax=ax, linewidth=LINEWIDTH)

    fig.suptitle("{}, {}".format(long_name, plot_units))

    with img_file.open("wb") as f:
        fig.savefig(f, bbox_inches="tight")

    plt.close(fig)



def plot_seasonal_integrated_water_vapour(samples_folder, seasons=commons.default_seasons):

    # TODO: make the integrations
    vname = "HU"
    file_prefix = "dm"

    plot_units = "kg/kg"
    mult_coeff = 1
    add_offset = 0

    out_dx = 0.5



    season_to_mean = OrderedDict()
    r = None
    for sname, months in seasons.items():

        # find files for the season
        paths = []
        for m in months:
            paths.extend(glob.glob(str(samples_folder.joinpath("*{:02d}/{}*".format(m, file_prefix)))))


        r = MultiRPN(paths)

        data = r.get_4d_field(varname=vname)
        season_to_mean[sname] = np.array([list(v.items())[0][1] for k, v in data.items()]).mean(axis=0)
        print("Processed: {}".format(sname))

    lons2d, lats2d = r.get_longitudes_and_latitudes_of_the_last_read_rec()



    img_folder = samples_folder.joinpath("images/seasonal")
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    img_file = img_folder.joinpath("{}.png".format(vname))

    plot_utils.apply_plot_params(width_cm=25, height_cm=15, font_size=18)
    fig = plt.figure()
    ncols = 2
    gs = GridSpec(len(season_to_mean) // ncols + int(not (len(season_to_mean) % ncols == 0)), ncols, wspace=0, hspace=0)
    xx, yy = None, None
    bmp = Basemap(projection="robin", lon_0=0)


    nlevs = 20
    clevs = None
    for i, (sname, field) in enumerate(season_to_mean.items()):
        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col])

        lons, lats, data_out = commons.interpolate_to_uniform_global_grid(field, lons_in=lons2d, lats_in=lats2d, out_dx=out_dx)

        if xx is None:
            xx, yy = bmp(lons, lats)


        cs = bmp.contourf(xx, yy, data_out * mult_coeff, nlevs if clevs is None else clevs, cmap=cm_basemap.s3pcpn_l, extend="max")

        # save color levels for next subplots
        clevs = cs.levels

        print(np.max(data_out * mult_coeff))

        ax.set_title(sname)
        cb = plt.colorbar(cs, ax=ax)
        if not (row == 0 and col == ncols - 1):
            # cb.ax.set_title(plot_units)
            cb.ax.set_visible(False)
        bmp.drawcoastlines(ax=ax, linewidth=LINEWIDTH)

    fig.suptitle("Precipitation, {}".format(plot_units))

    with img_file.open("wb") as f:
        fig.savefig(f, bbox_inches="tight")
    plt.close(fig)






@main_decorator
def main():
    samples_folder = "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/Global_NA_v1/Samples"
    samples_folder = Path(samples_folder)

    seasons = OrderedDict([
        ("Spring", (3, 4, 5)),
        ("Summer", (6, 7, 8)),
        ("Fall", (9, 10, 11)),
        ("Winter", (1, 2, 12))
    ])
    # plot_seasonal_precipitation(samples_folder, seasons=seasons)
    # plot_seasonal_2m_temp(samples_folder, seasons=seasons)
    # plot_seasonal_windspeed(samples_folder, seasons=seasons)
    plot_seasonal_vertical_velocity(samples_folder, seasons=seasons)


if __name__ == '__main__':
    main()
