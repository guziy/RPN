from netCDF4 import Dataset
from cartopy.feature import NaturalEarthFeature, GSHHSFeature
import collections
from iris.cube import Cube
from matplotlib import gridspec
from matplotlib.colors import LogNorm, BoundaryNorm
from pyhdf.V import V
from crcm5.analyse_hdf import common_plot_params as cpp
import my_colormaps

__author__ = 'huziy'

import os
import iris
import iris.quickplot as qplt
from iris import analysis, coord_categorisation
import matplotlib.pyplot as plt
import numpy as np
import cartopy
from matplotlib import cm

#EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP00/"
#EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP01/"
#EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_0.1deg/"

#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_Luis"

#EXP_DIR = "/skynet2_rech1/lduarte/NEMO/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_10KM/EXP00/"

#EXP_DIR = "/home/huziy/nemo_glk/test_fwb_my"
#EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_Luis_fwb2"

#EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP00"
#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/exp_0.1deg_from_restart_1958"
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

import nemo_commons

EXP_NAME = os.path.basename(EXP_DIR)
NEMO_IMAGES_DIR = os.path.join("nemo", EXP_NAME)

if not os.path.isdir(NEMO_IMAGES_DIR):
    os.mkdir(NEMO_IMAGES_DIR)


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
    fig = plt.figure(figsize=(7, 4))
    #fig.suptitle(cube.name() + " ({0})".format(cube.units))
    nplots = cube_seasonal.shape[0]
    ncols = 2
    nrows = nplots // ncols if nplots % ncols == 0 else nplots // ncols + 1

    b, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(T_FILE_PATH, resolution="i")
    x, y = b(lons, lats)
    gs = gridspec.GridSpec(ncols=ncols + 1, nrows=nrows, width_ratios=[1, 1, 0.05], wspace=0)  # +1 for the colorbar
    the_mask = nemo_commons.get_mask(path=os.path.join(EXP_DIR, "bathy_meter.nc"))


    vmin = None
    vmax = None

    for i, season in zip(range(nplots), cube_seasonal.coord("season").points):

        data = cube_seasonal.extract(iris.Constraint(season=season)).data
        the_min = data[the_mask].min()
        the_max = np.percentile(data[the_mask], 95)

        if vmin is None:
            vmin, vmax = the_min, the_max
        else:
            vmin = min(the_min, vmin)
            vmax = max(the_max, vmax)


    print "{0}: ".format(var_name), vmin, vmax
    cs = None
    for i, season in zip(range(nplots), cube_seasonal.coord("season").points):
        print season
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season.upper())
        data = cube_seasonal.extract(iris.Constraint(season=season)).data

        #plot only upper level of the 3d field if given
        if data.ndim > 2:
            if data.shape[1:] == x.shape:
                data = data[0, :, :]
            else:
                data = data[:, :, 0]

        to_plot = np.ma.masked_where(~the_mask, data)
        print to_plot.min(), to_plot.max()

        cs = b.pcolormesh(x, y, to_plot, ax=ax, vmin = vmin, vmax = vmax, cmap = cm.get_cmap("jet", 20))
        b.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)
        b.drawparallels(np.arange(-90, 90, 2))
        b.drawmeridians(np.arange(-180, 180, 2))

    plt.colorbar(cs, cax = fig.add_subplot(gs[:, ncols]))

    fname = "{0}_{1}.jpeg".format(cube_seasonal.var_name, "-".join(cube_seasonal.coord("season").points))
    if not os.path.isdir(NEMO_IMAGES_DIR):
        os.mkdir(NEMO_IMAGES_DIR)
    fig.tight_layout()
    fig.savefig(os.path.join(NEMO_IMAGES_DIR, fname), dpi=cpp.FIG_SAVE_DPI)





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
    b, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(T_FILE_PATH, resolution="h")
    x, y = b(lons, lats)
    the_mask = nemo_commons.get_mask(path=os.path.join(EXP_DIR, "bathy_meter.nc"))

    levels = np.arange(0, 0.11, 0.01)
    bn = BoundaryNorm(levels, len(levels) - 1)
    cmap = cm.get_cmap("Accent", len(levels) - 1)


    for season in u_cube_seasonal.coord("season").points:
        print season
        fig = plt.figure(figsize=(8,4))

        ax = fig.add_subplot(111)
        ax.set_title(season.upper())
        u = u_cube_seasonal.extract(iris.Constraint(season=season)).data
        v = v_cube_seasonal.extract(iris.Constraint(season=season)).data

        u = np.ma.masked_where(~the_mask, u)
        v = np.ma.masked_where(~the_mask, v)

        speed = np.sqrt(u ** 2 + v ** 2)
        cs = b.pcolormesh(x, y, speed, norm = bn, vmin=levels[0], vmax = levels[-1], cmap = cmap)
        b.colorbar(cs)

        u, v = b.rotate_vector(u, v, lons, lats)
        q = b.quiver(x, y, u, v, scale = 1.5, width = 0.002)

        qk = plt.quiverkey(q, 0.15, 0.1, 0.05, '0.05 m/s', labelpos='W')
        b.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)


        fname = "{0}-{1}_{2}.jpeg".format(u_cube_seasonal.var_name, v_cube_seasonal.var_name, season)

        if not os.path.isdir(NEMO_IMAGES_DIR):
            os.mkdir(NEMO_IMAGES_DIR)
        fig.tight_layout()
        fig.savefig(os.path.join(NEMO_IMAGES_DIR, fname), dpi=cpp.FIG_SAVE_DPI)

    #plot annual mean
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fname = "{0}-{1}_{2}.jpeg".format(u_cube_seasonal.var_name, v_cube_seasonal.var_name, "annual")
    u_annual = u_cube_seasonal.collapsed("season", analysis.MEAN).data
    v_annual = v_cube_seasonal.collapsed("season", analysis.MEAN).data

    u_annual = np.ma.masked_where(~the_mask, u_annual)
    v_annual = np.ma.masked_where(~the_mask, v_annual)

    fig.suptitle("Annual")
    q = b.quiver(x, y, u_annual, v_annual, scale = 1.5, width = 0.002, zorder = 5)
    qk = plt.quiverkey(q, 0.15, 0.1, 0.05, '0.05 m/s', labelpos='W')


    levels = np.arange(0, 0.15, 0.01)
    bn = BoundaryNorm(levels, len(levels) - 1)
    #cmap = my_colormaps.get_cmap_from_ncl_spec_file("colormap_files/wgne15.rgb", len(levels) - 1)
    cmap = cm.get_cmap("Paired", len(levels) - 1)
    cs = b.pcolormesh(x, y, np.sqrt(u_annual ** 2 + v_annual ** 2), cmap = cmap, norm = bn)
    b.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)
    b.colorbar(cs)
    fig.tight_layout()
    fig.savefig(os.path.join(NEMO_IMAGES_DIR, fname), dpi=cpp.FIG_SAVE_DPI)


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()

    draw_seasonal_means_panel(path=T_FILE_PATH)
    # draw_seasonal_means_panel(path=T_FILE_PATH, var_name="sossheig")
    # draw_seasonal_means_panel(path=T_FILE_PATH, var_name="somixhgt")
    # draw_seasonal_means_panel(path=T_FILE_PATH, var_name="somxl010")
    draw_seasonal_means_panel(path=T_FILE_PATH, var_name="soicecov")
    # draw_seasonal_means_panel(path=T_FILE_PATH, var_name="sowindsp")


    #zonal and meridional currents
    # draw_seasonal_means_panel(path=U_FILE_PATH, var_name="vozocrtx")
    # draw_seasonal_means_panel(path=V_FILE_PATH, var_name="vomecrty")


    plot_vector_fields(u_path=U_FILE_PATH, v_path=V_FILE_PATH)
    #draw_for_date()