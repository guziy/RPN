from netCDF4 import Dataset
from cartopy.feature import NaturalEarthFeature, GSHHSFeature
import collections
from iris.cube import Cube
from matplotlib import gridspec
from matplotlib.colors import LogNorm, BoundaryNorm
from pyhdf.V import V
from crcm5.analyse_hdf import common_plot_params as cpp

__author__ = 'huziy'

import os
import iris
import iris.quickplot as qplt
from iris import analysis, coord_categorisation
import matplotlib.pyplot as plt
import numpy as np
import cartopy

EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP00/"

T_FILE_PATH = EXP_DIR + "GLK_1d_19580101_19581231_grid_T.nc"
U_FILE_PATH = EXP_DIR + "GLK_1d_19580101_19581231_grid_U.nc"
V_FILE_PATH = EXP_DIR + "GLK_1d_19580101_19581231_grid_V.nc"

import nemo_commons

NEMO_IMAGES_DIR = "nemo"


def draw_for_date():
    cubes = iris.load(T_FILE_PATH)
    sst = cubes[1]

    coord_categorisation.add_month_number(sst, "time")
    coord_categorisation.add_day_of_month(sst, "time")

    sst_sel = sst.extract(iris.Constraint(month_number=7, day_of_month=1))
    sst_sel.data = np.ma.masked_where(sst_sel.data == 0, sst_sel.data)

    b, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(T_FILE_PATH)

    #Plot the data
    fig = plt.figure()

    x, y = b(lons, lats)
    img = b.pcolormesh(x, y, sst_sel.data)
    b.colorbar(img)
    b.drawcoastlines()
    fname = "sst_1july_1958.jpeg"
    if not os.path.isdir(NEMO_IMAGES_DIR):
        os.mkdir(NEMO_IMAGES_DIR)
    fig.savefig(os.path.join(NEMO_IMAGES_DIR, fname))


def draw_seasonal_means_panel(path="", var_name="sosstsst"):
    cube = iris.load_cube(path, constraint=iris.Constraint(cube_func=lambda c: c.var_name == var_name))
    assert isinstance(cube, Cube)


    #Add month_number coordinate
    #coord_categorisation.add_month_number(cube, "time")
    coord_categorisation.add_season(cube, "time")

    cube_seasonal = cube.aggregated_by("season", analysis.MEAN)

    print cube_seasonal.shape

    #plot results
    fig = plt.figure()
    fig.suptitle(cube.name() + " ({0})".format(cube.units))
    nplots = cube_seasonal.shape[0]
    ncols = 2
    nrows = nplots // ncols if nplots % ncols == 0 else nplots // ncols + 1

    b, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(T_FILE_PATH)
    x, y = b(lons, lats)
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows)  # +1 for the colorbar
    the_mask = nemo_commons.get_mask()

    for i, season in zip(range(nplots), cube_seasonal.coord("season").points):
        print season
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season.upper())
        data = cube_seasonal.extract(iris.Constraint(season=season)).data

        #plot only upper level of the 3d field if given
        if len(data.shape) > 2:
            if data.shape[1:] == x.shape:
                data = data[0, :, :]
            else:
                data = data[:, :, 0]

        to_plot = np.ma.masked_where(~the_mask, data)
        cs = b.contourf(x, y, to_plot, ax=ax, levels=nemo_commons.varname_to_colorlevels.get(var_name, None))
        b.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)
        b.drawparallels(np.arange(-90, 90, 10))
        b.drawmeridians(np.arange(-180, 180, 10))
        b.colorbar(cs)

    fname = "{0}_{1}.jpeg".format(cube_seasonal.var_name, "-".join(cube_seasonal.coord("season").points))
    if not os.path.isdir(NEMO_IMAGES_DIR):
        os.mkdir(NEMO_IMAGES_DIR)
    fig.tight_layout()
    fig.savefig(os.path.join(NEMO_IMAGES_DIR, fname), bbox_inches="tight", dpi=cpp.FIG_SAVE_DPI)





def plot_vector_fields(u_path="", v_path="", u_name="vozocrtx", v_name="vomecrty", level=0):
    name_constraint = iris.Constraint(cube_func=lambda c: c.var_name == u_name or c.var_name == v_name)

    u_cube = iris.load_cube(u_path, constraint=name_constraint)
    u_cube = u_cube.extract(iris.Constraint(model_level_number = u_cube.coord("model_level_number").points[level]))

    v_cube = iris.load_cube(v_path, constraint=name_constraint)
    v_cube = v_cube.extract(iris.Constraint(model_level_number = v_cube.coord("model_level_number").points[level]))
    assert isinstance(u_cube, Cube)

    #calculate seasonal means
    coord_categorisation.add_season(u_cube, "time")
    coord_categorisation.add_season(v_cube, "time")

    u_cube_seasonal = u_cube.aggregated_by("season", analysis.MEAN)
    v_cube_seasonal = v_cube.aggregated_by("season", analysis.MEAN)



    #plot results
    b, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(T_FILE_PATH)
    x, y = b(lons, lats)
    the_mask = nemo_commons.get_mask()

    levels = np.arange(0, 0.24, 0.004)
    bn = BoundaryNorm(levels, len(levels) - 1)


    for season in u_cube_seasonal.coord("season").points:
        print season
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.set_title(season.upper())
        u = u_cube_seasonal.extract(iris.Constraint(season=season)).data
        v = v_cube_seasonal.extract(iris.Constraint(season=season)).data

        u = np.ma.masked_where(~the_mask, u)
        v = np.ma.masked_where(~the_mask, v)

        speed = u ** 2 + v ** 2
        cs = b.pcolormesh(x, y, speed, norm = bn, vmin=levels[0], vmax = levels[-1])
        b.colorbar(cs)

        q = b.quiver(x[::10, ::10], y[::10, ::10], u[::10, ::10], v[::10, ::10], scale = 1, linewidth = 0.1)
        qk = plt.quiverkey(q, 0.25, 0.1, 0.2, '0.2 m/s', labelpos='W')
        b.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)


        fname = "{0}-{1}_{2}.jpeg".format(u_cube_seasonal.var_name, v_cube_seasonal.var_name, season)

        if not os.path.isdir(NEMO_IMAGES_DIR):
            os.mkdir(NEMO_IMAGES_DIR)
        fig.tight_layout()
        fig.savefig(os.path.join(NEMO_IMAGES_DIR, fname), bbox_inches="tight", dpi=cpp.FIG_SAVE_DPI)

    #plot annual mean
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fname = "{0}-{1}_{2}.jpeg".format(u_cube_seasonal.var_name, v_cube_seasonal.var_name, "annual")
    u_annual = u_cube_seasonal.collapsed("season", analysis.MEAN).data
    v_annual = v_cube_seasonal.collapsed("season", analysis.MEAN).data

    u_annual = np.ma.masked_where(~the_mask, u_annual)
    v_annual = np.ma.masked_where(~the_mask, v_annual)

    fig.suptitle("Annual")
    b.quiver(x[::5, ::5], y[::5, ::5], u_annual[::5, ::5], v_annual[::5, ::5], scale = 1, linewidth = 0.01)
    b.drawcoastlines()
    fig.tight_layout()
    fig.savefig(os.path.join(NEMO_IMAGES_DIR, fname), bbox_inches="tight", dpi=cpp.FIG_SAVE_DPI)


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()

    #draw_seasonal_means_panel(path=T_FILE_PATH)
    #draw_seasonal_means_panel(path=T_FILE_PATH, var_name="sossheig")


    #zonal and meridional currents
    # draw_seasonal_means_panel(path=U_FILE_PATH, var_name="vozocrtx")
    # draw_seasonal_means_panel(path=V_FILE_PATH, var_name="vomecrty")

    plot_vector_fields(u_path=U_FILE_PATH, v_path=V_FILE_PATH)
    #draw_for_date()