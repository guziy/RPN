from datetime import datetime
import iris
from iris.cube import Cube
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter
from matplotlib.gridspec import GridSpec

import iris.quickplot as iplt
from matplotlib.patches import Rectangle

__author__ = 'huziy'

import os
import nemo_commons
import matplotlib.pyplot as plt
import application_properties
import numpy as np

application_properties.set_current_directory()

#EXP_DIR = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_0.1deg/"
#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_Luis"
#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/EXP_Luis_fwb2"
#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/exp_Luis_100year_clim"

EXP_DIR = "/home/huziy/nemo_glk/test_fwb_my"
#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/exp_0.1deg_from_restart_1958"


#EXP_DIR = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_0.1deg_lim3"
T_FILE_PATH, U_FILE_PATH, V_FILE_PATH = None, None, None

for fname in os.listdir(EXP_DIR):
    if fname.endswith("_grid_T.nc"):
        T_FILE_PATH = os.path.join(EXP_DIR, fname)
    elif fname.endswith("_grid_U.nc"):
        U_FILE_PATH = os.path.join(EXP_DIR, fname)
    elif fname.endswith("_grid_V.nc"):
        V_FILE_PATH = os.path.join(EXP_DIR, fname)

EXP_NAME = os.path.basename(EXP_DIR)
NEMO_IMAGES_DIR = os.path.join("nemo_1d", EXP_NAME)

if not os.path.isdir(NEMO_IMAGES_DIR):
    os.mkdir(NEMO_IMAGES_DIR)


def draw_timeseries(path=T_FILE_PATH, var_name="sosstsst", nx=1, ny=1,
                    lower_left_ij=None, basemap=None, x2d=None, y2d=None,
                    the_mask=None):
    """
    Note: assumes the following order of the field dimensions

    time, z, y, x

    :param path:
    :param var_name:
    :param nx:
    :param ny:
    :param lower_left_ij:
    :param basemap:
    :param x2d:
    :param y2d:
    """
    cube = iris.load_cube(path, constraint=iris.Constraint(cube_func=lambda c: c.var_name == var_name))
    assert isinstance(cube, Cube)
    print cube.shape
    print lower_left_ij
    subcube = cube[:, lower_left_ij[1]:lower_left_ij[1] + ny, lower_left_ij[0]:lower_left_ij[0] + nx]


    mult = the_mask.astype(int)
    data = cube.data * mult[np.newaxis, :, :]

    subdata = data[:, lower_left_ij[1]:lower_left_ij[1] + ny, lower_left_ij[0]:lower_left_ij[0] + nx]


    #subcube.collapsed(["longitude", "latitude"], iris.analysis.MEAN)

    fig = plt.figure()
    gs = GridSpec(2, 1)
    ax = fig.add_subplot(gs[0, 0])

    #subcube = subcube[:, 0, 0]
    ax.plot(np.sum(subdata, axis=(1, 2)))
    #ax.xaxis.set_major_formatter(DateFormatter("%Y\n%m-%d"))
    ax.set_title(var_name)

    ax = fig.add_subplot(gs[1, 0])
    data = cube.collapsed(["time"], iris.analysis.MEAN).data
    if the_mask is not None:
        data = np.ma.masked_where(~the_mask, data)
    im = basemap.pcolormesh(x2d, y2d, data, ax=ax)
    basemap.colorbar(im)
    basemap.drawcoastlines(linewidth=0.5)

    w, h = x2d[lower_left_ij[1] + ny - 1, lower_left_ij[0] + nx - 1] - x2d[lower_left_ij[1], lower_left_ij[0]], \
           y2d[lower_left_ij[1] + ny - 1, lower_left_ij[0]] - y2d[lower_left_ij[1], lower_left_ij[0]]

    print w, h

    r = Rectangle((x2d[lower_left_ij[1], lower_left_ij[0]],
                   y2d[lower_left_ij[1], lower_left_ij[0]]), w, h, facecolor="none")
    ax.add_patch(r)

    im_path = os.path.join(NEMO_IMAGES_DIR, "ts_{4}_{0}_{1}_{2}_{3}.jpeg".format(lower_left_ij[0],
                                                                                 lower_left_ij[1],
                                                                                 nx, ny, var_name))
    if not os.path.isdir(NEMO_IMAGES_DIR):
        os.mkdir(NEMO_IMAGES_DIR)

    fig.tight_layout()
    fig.savefig(im_path)


def main():
    b, lons2d, lats2d = nemo_commons.get_basemap_and_coordinates_from_file(
        path=os.path.join(EXP_DIR, "bathy_meter.nc"), resolution="i")
    x2d, y2d = b(lons2d, lats2d)

    the_mask = nemo_commons.get_mask(path=os.path.join(EXP_DIR, "bathy_meter.nc"))

#Ontario
    lower_left_indices = (123, 20)
    dix, djy = 40, 14

#Erie
#    lower_left_indices = (88, 2)
#    dix, djy = 50, 18

#Huron
#    lower_left_indices = (75, 17)
#    dix, djy = 49, 36


#    lower_left_indices = (115, 5)
#    dix, djy = 15, 10

#Superior
    lower_left_indices = (0, 53)
    dix, djy = 85, 30



    draw_timeseries(path=T_FILE_PATH, lower_left_ij=lower_left_indices, nx=dix, ny=djy,
                    var_name="sossheig", basemap=b, x2d=x2d, y2d=y2d, the_mask=the_mask)

    # draw_timeseries(path=T_FILE_PATH, lower_left_ij=lower_left_indices, nx=dix, ny=djy,
    #                 var_name="sowaflup", basemap=b, x2d=x2d, y2d=y2d, the_mask=the_mask)
    #
    # draw_timeseries(path=T_FILE_PATH, lower_left_ij=lower_left_indices, nx=dix, ny=djy,
    #                 var_name="soicecov", basemap=b, x2d=x2d, y2d=y2d, the_mask=the_mask)
    #
    # draw_timeseries(path=T_FILE_PATH, lower_left_ij=lower_left_indices, nx=dix, ny=djy,
    #                 var_name="sosstsst", basemap=b, x2d=x2d, y2d=y2d, the_mask=the_mask)

    plt.show()

    pass


if __name__ == "__main__":
    main()