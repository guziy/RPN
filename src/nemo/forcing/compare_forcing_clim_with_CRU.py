from collections import OrderedDict
from datetime import datetime
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from scipy.spatial.ckdtree import cKDTree
from crcm5 import infovar
from cru.temperature import CRUDataManager
from dfs_data_manager import DFSDataManager
from util.geo import lat_lon
import os

__author__ = 'huziy'

#Compare seasonal climatological means of the forcing data (DFS) and CRU observed data
import matplotlib.pyplot as plt
import numpy as np

NEMO_IMAGES_DIR = os.path.join("nemo", "forcing_dfs_5.2")

from nemo import nemo_commons


def plot_errors_in_one_figure(season_to_diff, fig_path="", **kwargs):
    fig = plt.figure()
    basemap = kwargs["basemap"]
    x, y = kwargs["x"], kwargs["y"]

    vmin, vmax = None, None
    for k, field in season_to_diff.iteritems():
        pl, ph = field.min(), field.max()
        if vmin is None:
            vmin = pl
            vmax = ph
        else:
            vmin = min(vmin, pl)
            vmax = max(vmax, ph)

    print "min,max = ({0}, {1})".format(vmin, vmax)
    ncolors = 25
    if vmin * vmax <= 0:
        cmap = cm.get_cmap("RdBu_r", ncolors)
    else:
        if vmin >= 0:
            cmap = cm.get_cmap("Reds", ncolors)
        else:
            cmap = cm.get_cmap("Blues_r", ncolors)

    bn, bounds, vmin, vmax = infovar.get_boundary_norm(vmin, vmax, ncolors, exclude_zero=True, difference=True)

    gs = GridSpec(len(season_to_diff), 2, width_ratios=[1, 0.05])
    row = 0
    img = None
    for the_seasson, the_diff in season_to_diff.iteritems():
        ax = fig.add_subplot(gs[row, 0])
        #img = basemap.pcolormesh(x, y, the_diff, ax = ax, norm = bn, cmap = cmap, vmin = vmin, vmax = vmax)
        img = basemap.contourf(x, y, the_diff, ax = ax, norm = bn, cmap = cmap, extend = "both", levels = bounds)
        ax.set_title(the_seasson)
        basemap.drawcoastlines()
        row += 1
    cax = fig.add_subplot(gs[:, 1])


    plt.colorbar(img,ticks = bounds, orientation = "vertical", cax = cax, extend = "both")
    fig.savefig(fig_path, bbox_inches = "tight")


def main(dfs_var_name="t2", cru_var_name="tmp",
         dfs_folder="/home/huziy/skynet3_rech1/NEMO_OFFICIAL/DFS5.2_interpolated",
         cru_file = "data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.tmp.dat.nc"):

    if not os.path.isdir(NEMO_IMAGES_DIR):
        os.mkdir(NEMO_IMAGES_DIR)

    #year range is inclusive [start_year, end_year]
    start_year = 1981
    end_year = 2009

    season_name_to_months = OrderedDict([
        ("Winter", (1, 2, 12)),
        ("Spring", range(3, 6)),
        ("Summer", range(6, 9)),
        ("Fall", range(9, 12))])

    cru_t_manager = CRUDataManager(var_name=cru_var_name, path=cru_file)
    cru_lons, cru_lats = cru_t_manager.lons2d, cru_t_manager.lats2d
    #get seasonal means (CRU)
    season_to_mean_cru = cru_t_manager.get_seasonal_means(season_name_to_months=season_name_to_months,
                                                          start_year=start_year,
                                                          end_year=end_year)
    #get seasonal means Drakkar
    dfs_manager = DFSDataManager(folder_path=dfs_folder, var_name=dfs_var_name)
    season_to_mean_dfs = dfs_manager.get_seasonal_means(season_name_to_months=season_name_to_months,
                                                        start_year=start_year,
                                                        end_year=end_year)

    dfs_lons, dfs_lats = dfs_manager.get_lons_and_lats_2d()
    xt, yt, zt = lat_lon.lon_lat_to_cartesian(dfs_lons.flatten(), dfs_lats.flatten())
    xs, ys, zs = lat_lon.lon_lat_to_cartesian(cru_lons.flatten(), cru_lats.flatten())
    ktree = cKDTree(data=zip(xs, ys, zs))
    dists, inds = ktree.query(zip(xt, yt, zt))

    season_to_err = OrderedDict()
    for k in season_to_mean_dfs:
        interpolated_cru = season_to_mean_cru[k].flatten()[inds].reshape(dfs_lons.shape)
        if dfs_var_name.lower() == "t2":
            #interpolated_cru += 273.15
            season_to_mean_dfs[k] -= 273.15
        elif dfs_var_name.lower() == "precip":  # precipitation in mm/day
            season_to_mean_dfs[k] *= 24 * 60 * 60

        season_to_err[k] = season_to_mean_dfs[k] #- interpolated_cru

    season_indicator = "-".join(sorted(season_to_err.keys()))
    fig_path = os.path.join(NEMO_IMAGES_DIR, "{3}_errors_{0}-{1}_{2}_dfs.jpeg".format(start_year,
                                                                                  end_year,
                                                                                  season_indicator,
                                                                                  dfs_var_name))

    basemap = nemo_commons.get_default_basemap_for_glk(dfs_lons, dfs_lats, resolution="l")
    x, y = basemap(dfs_lons, dfs_lats)
    coords_and_basemap = {
        "basemap": basemap, "x": x, "y": y
    }

    plot_errors_in_one_figure(season_to_err, fig_path=fig_path, **coords_and_basemap)


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    main(dfs_var_name="precip", cru_var_name="pre",
         cru_file="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc")