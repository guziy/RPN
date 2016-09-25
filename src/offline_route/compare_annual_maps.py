from collections import OrderedDict

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec

from application_properties import main_decorator
from util import plot_utils

import matplotlib.pyplot as plt

from netCDF4 import Dataset

from crcm5.mh_domains import default_domains

import numpy as np


def plot_annual_mean(ax, bmp, lons, lats, data, x_index, y_index, norm=None, colormap=None):
    # calculate annual mean

    the_mean = data.mean(axis=0)
    to_plot = np.ma.masked_all_like(lons)
    to_plot[x_index, y_index] = the_mean


    # do the plotting
    xx, yy = bmp(lons, lats)
    im = bmp.pcolormesh(xx, yy, to_plot, norm=norm, cmap=colormap, ax=ax)
    bmp.colorbar(im, ax=ax)
    bmp.drawcoastlines(ax=ax)



@main_decorator
def main():

    shape_path = default_domains.MH_BASINS_PATH


    label_to_sim_path = OrderedDict(
        [("0.11", "/RESCUE/skynet3_rech1/huziy/water_route_mh_bc_011deg_wc/discharge_1980_01_01_12_00.nc"),
         ("0.44", "/RESCUE/skynet3_rech1/huziy/water_route_mh_bc_044deg_wc/discharge_1980_01_01_12_00.nc")]
    )

    label_to_gridconfig = {
        "0.11": default_domains.bc_mh_011,
        "0.44": default_domains.gc_cordex_na_044
    }


    label_to_infocell = {
        "0.11": "/RESCUE/skynet3_rech1/huziy/water_route_mh_bc_011deg_wc/infocell.nc",
        "0.44": "/RESCUE/skynet3_rech1/huziy/water_route_mh_bc_044deg_wc/infocell.nc"
    }


    # setup subplots
    plot_utils.apply_plot_params()
    nrows = 1
    ncols = len(label_to_sim_path)

    gs = GridSpec(nrows=nrows, ncols=ncols)
    fig = plt.figure()

    clevs = [0, 20, 50, 100, 200, 500, 1000, 1500, 3000, 4500, 5000, 7000, 9000]
    bn = BoundaryNorm(clevs, len(clevs) - 1)
    cmap = cm.get_cmap("jet", len(clevs) - 1)

    for col, (label, the_path) in enumerate(label_to_sim_path.items()):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(label)

        with Dataset(the_path) as ds:
            data, x_index, y_index = [ds.variables[k][:] for k in ["water_discharge", "x_index", "y_index"]]


        with Dataset(label_to_infocell[label]) as ds:
            lon, lat = [ds.variables[k][:] for k in ["lon", "lat"]]

        gc = label_to_gridconfig[label]
        # get the basemap object
        bmp, data_mask = gc.get_basemap_using_shape_with_polygons_of_interest(
            lon, lat, shp_path=shape_path, mask_margin=5)

        plot_annual_mean(ax, bmp, lons=lon, lats=lat, data=data, x_index=x_index, y_index=y_index, norm=bn, colormap=cmap)

    fig.savefig("mh/offline_route_annual_mean_comparison.png", bbox_inches="tight")



if __name__ == '__main__':
    main()