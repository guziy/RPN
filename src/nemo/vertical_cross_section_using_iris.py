from netCDF4 import Dataset
import os
from geopy.distance import GreatCircleDistance
import iris
from iris import coord_categorisation, analysis
from matplotlib import gridspec
from matplotlib.axes import Axes

from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D
from crcm5.analyse_hdf import common_plot_params as cpp
import numpy as np
from matplotlib import cm
from util import plot_utils

__author__ = 'huziy'


import matplotlib.pyplot as plt


#EXP_DIR = "/home/huziy/nemo_glk/test_fwb_my"
#EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_Luis_fwb2"
#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_0.1deg_lim3"
#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/Simulations/1981-2000_Sim_per_lake_100yr_spinup_LIM3/Huron"
EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/Simulations/1981-2000_Sim_per_lake_100yr_spinup_LIM3/Huron"

T_FILE_PATH, U_FILE_PATH, V_FILE_PATH = None, None, None

import application_properties
application_properties.set_current_directory()

for fname in os.listdir(EXP_DIR):
    if fname.endswith("_grid_T.nc"):
        T_FILE_PATH = os.path.join(EXP_DIR, fname)
    elif fname.endswith("_grid_U.nc"):
        U_FILE_PATH = os.path.join(EXP_DIR, fname)
    elif fname.endswith("_grid_V.nc"):
        V_FILE_PATH = os.path.join(EXP_DIR, fname)

from . import nemo_commons

EXP_NAME = os.path.basename(EXP_DIR)
NEMO_IMAGES_DIR = os.path.join("nemo", EXP_NAME)

if not os.path.isdir(NEMO_IMAGES_DIR):
    os.mkdir(NEMO_IMAGES_DIR)



def get_section_hor_indices(i_start = 0, j_start = 0, i_end = -1, j_end = -1):

    assert i_end >= 0 and j_end >= 0

    npoints = max(abs(i_end - i_start), abs(j_end - j_start)) + 1
    dj = (j_end - j_start) / float(npoints - 1)
    j_list = [j_start + int(k * dj) for k in range(npoints)]

    di = (i_end - i_start) / float(npoints - 1)
    i_list = [i_start + int(di * k) for k in range(npoints)]

    assert len(i_list) == len(j_list)
    print(i_list, j_list)
    return np.array(i_list, dtype=int), np.array(j_list, dtype=int)


def plot_cross_section_for_seasons(data_path = "", i_start = 0, j_start = 0, i_end = -1, j_end = -1,
                                   var_name = None):

    name_constraint = iris.Constraint(cube_func=lambda c: c.var_name == var_name)
    data_cube = iris.load_cube(data_path, constraint=name_constraint)


    fig = plt.figure()
    impath = os.path.join(NEMO_IMAGES_DIR, "vert_sect_{0}_{1}_{2}_{3}_{4}.jpeg".format(var_name,
                                                                                       i_start, j_start,
                                                                                       i_end, j_end))


       #Add month_number coordinate
    #coord_categorisation.add_month_number(cube, "time")
    coord_categorisation.add_season(data_cube, "time")

    cube_seasonal = data_cube.aggregated_by("season", analysis.MEAN)

    nplots = cube_seasonal.shape[0]
    ncols = 2
    nrows = nplots // ncols if nplots % ncols == 0 else nplots // ncols + 1

    b, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(T_FILE_PATH, resolution="i")

    print("lons shape: ", lons.shape)

    gs = gridspec.GridSpec(ncols=ncols + 1, nrows=nrows + 1, width_ratios=[1, 1, 0.05], hspace=0.3)  # +1 for the colorbar and for map

    bath_path = os.path.join(EXP_DIR, "bathy_meter.nc")
    the_mask = nemo_commons.get_mask(path=bath_path)

    depths = Dataset(T_FILE_PATH).variables["deptht"][:]
    bathymetry = Dataset(bath_path).variables["Bathymetry"][:]

    vert_mask = np.ma.masked_all(depths.shape + bathymetry.shape)
    for lev, di in enumerate(depths):
        not_mask_j, not_mask_i = np.where(di < bathymetry * 0.95)
        vert_mask[lev, not_mask_j, not_mask_i] = 1

    lons_sel, lats_sel = None, None
    depths_2d = None
    vmin = None
    vmax = None
    nx, ny = None, None
    dists_2d = None
    season_to_section = {}
    for i, season in zip(list(range(nplots)), cube_seasonal.coord("season").points):

        data = cube_seasonal.extract(iris.Constraint(season=season)).data.squeeze()

        assert data.ndim == 3
        _, ny, nx = data.shape

        if i_end == -1:
            i_end = nx - 1
        if j_end == -1:
            j_end = ny - 1

        j_mask, i_mask = np.where(~the_mask)
        print("mask shape: ", the_mask.shape)
        print("data shape: ", data.shape)
        data[:, j_mask, i_mask] = np.ma.masked

        i_list, j_list = get_section_hor_indices(i_start=i_start, i_end = i_end, j_start=j_start, j_end=j_end)



        data_sel = data[:, j_list, i_list]
        data_sel = np.ma.masked_where(vert_mask[:, j_list, i_list].mask, data_sel)
        print("data_sel shape: ", data_sel.shape)
        if lons_sel is None:
            lons_sel = lons[j_list, i_list]
            lats_sel = lats[j_list, i_list]


            p_start = (lats[j_start, i_start], lons[j_start, i_start])
            dists = [GreatCircleDistance(p_start, (the_lat, the_lon)).km
                     for the_lat, the_lon in zip(lats_sel, lons_sel)]

            dists_2d, depths_2d = np.meshgrid(dists, depths)

        season_to_section[season] = data_sel

        if vmin is None:
            vmin = data_sel.min()
        else:
            vmin = min(vmin, data_sel.min())

        if vmax is None:
            vmax = data_sel.max()
        else:
            vmax = max(vmax, data_sel.max())

    delta = 1.0
    clevs = np.arange(np.floor(vmin), vmax + delta, delta)
    cmap = cm.get_cmap("jet", len(clevs) - 1)
    if clevs is not None:
        bn = BoundaryNorm(clevs, len(clevs) - 1)
    else:
        bn = None

    print("{0}: ".format(var_name), vmin, vmax)
    cs = None
    ax = None
    for i, season in zip(list(range(nplots)), cube_seasonal.coord("season").points):
        print(season)
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])

        ax.set_title(season.upper())
        data = season_to_section[season]
        print(data.min(), data.max())


        #to_plot = np.ma.masked_where(~the_mask, data)
        #print to_plot.min(), to_plot.max()

        #cs = ax.pcolormesh(dists_2d, depths_2d, data, norm = bn, cmap = cmap)
        cs = ax.contourf(dists_2d, depths_2d, data, levels = clevs, cmap = cmap)
        #cs = ax.pcolormesh(dists_2d, depths_2d, data, norm = bn, cmap = cmap)
        #b.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)
        #b.drawparallels(np.arange(-90, 90, 2))
        #b.drawmeridians(np.arange(-180, 180, 2))

        if col != 0:
            ax.yaxis.set_ticks([])

        ax.xaxis.set_ticks([])

        assert isinstance(ax, Axes)
        ax.invert_yaxis()
        ax.set_ylim(80, 0)  # Disregard areas deeper than 150 m
        if row == 0 and col == 0:
            ax.set_ylabel("Depth (m)", fontdict={"fontsize": 20})
    cb = plt.colorbar(cs, ticks = clevs[::2], cax = fig.add_subplot(gs[:nrows, ncols]))



    ax = fig.add_subplot(gs[nrows, :])
    x, y = b(lons, lats)

    b.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)
    b.fillcontinents()


    assert isinstance(ax, Axes)
    ax.add_line(Line2D([x[j_start, i_start], x[j_end, i_end]],
                        [y[j_start, i_start], y[j_end, i_end]], linewidth=3))

    fig.savefig(impath, bbox_inches = "tight")




    pass


def main():
    plot_utils.apply_plot_params(font_size=20, width_pt=None, width_cm=20, height_cm=20)
    i_start, j_start = 0, 0
    i_end, j_end = -1, -1

    print(get_section_hor_indices(i_start=5, j_start = 0, i_end = 10, j_end = 20))


    # #Superior
    # params = dict(
    #     i_start = 25, j_start = 65,
    #     i_end = 70, j_end = 65,
    #     var_name = "votemper"
    #
    # )
    # plot_cross_section_for_seasons(data_path=T_FILE_PATH, **params)
    #
    # #Michigan
    # params = dict(
    #     i_start = 55, j_start = 55,
    #     i_end = 55, j_end = 5,
    #     var_name = "votemper"
    #
    # )
    # plot_cross_section_for_seasons(data_path=T_FILE_PATH, **params)

    #Huron
    params = dict(
        i_start = 10, j_start = 30,
        i_end = 30, j_end = 10,
        var_name = "votemper"

    )
    plot_cross_section_for_seasons(data_path=T_FILE_PATH, **params)
    plt.show()



if __name__ == "__main__":
    main()