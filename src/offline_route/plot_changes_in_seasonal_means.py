from datetime import datetime
from netCDF4 import Dataset
import pickle
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
import os
from matplotlib.transforms import Affine2D
import pandas
from rpn.rpn import RPN
import scipy
from scipy.stats import stats
import my_colormaps
from .plot_seasonal_means import TIME_FORMAT, get_arctic_basemap, get_arctic_basemap_nps

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def get_land_sea_glaciers_mask_from_geophysics_file(
        path="/b10_fs1/winger/Arctic/OMSC26_Can_long_new_v01/Geophys/land_sea_glacier_mask_free"):

    r = RPN(path)
    mask = r.get_first_record_for_name("FMSK") < 0.5
    r.close()
    return mask


def main(months=None, season="DJF", ax = None, clevels = None,
         labels = None, paths = None):
    if not months:
        months = [12, 1, 2]

    path_to_glaciers_land_sea_mask = "/b2_fs2/huziy/geophy_from_others/land_sea_glacier_mask_phy"

    land_sea_glaciers_mask = get_land_sea_glaciers_mask_from_geophysics_file(path=path_to_glaciers_land_sea_mask)
    p_current = "{0}-{1}".format(start_year_current, end_year_current)
    p_future = "{0}-{1}".format(start_year_future, end_year_future)

    lons2d = None
    lats2d = None

    x_index = None
    y_index = None
    mean_data = None

    b = None

    for the_path, label in zip(paths, labels):
        ds = Dataset(the_path)
        if lons2d is None:
            lons2d = ds.variables["longitude"][:]
            lats2d = ds.variables["latitude"][:]

            #b = get_arctic_basemap(lons2d, lats2d)
            b = get_arctic_basemap_nps(round = True)

            x_index = ds.variables["x_index"][:]
            y_index = ds.variables["y_index"][:]

        cache_file = "_".join([str(m) for m in months]) + "_{0}-{1}_{2}-{3}_{4}_mean_change_cache.bin".format(
            start_year_current, end_year_current, start_year_future, end_year_future, label)


        #os.remove(cache_file)
        if not os.path.isfile(cache_file):
            time_str = ds.variables["time"][:]
            times = [datetime.strptime("".join(t_s), TIME_FORMAT) for t_s in time_str]

            data = ds.variables["water_discharge_accumulated"][:]
            df = pandas.DataFrame(data=data, index=times)
            df["year"] = df.index.map(lambda d: d.year)
            df["month"] = df.index.map(lambda d: d.month)

            print(df.shape, df.columns)

            data_current = df.ix[
                           (df.year >= start_year_current) & (df.year <= end_year_current) & df.month.isin(months), :]
            print(data_current.columns)
            data_current = data_current.drop(["year", "month"], axis=1)
            seasonal_means_current = data_current.groupby(
                by=lambda d: d.year).mean()  #calculate mean for the season for each year

            data_future = df.ix[(df.year >= start_year_future) & (df.year <= end_year_future) & df.month.isin(months),
                          :]
            data_future = data_future.drop(["year", "month"], axis=1)
            seasonal_means_future = data_future.groupby(by=lambda d: d.year).mean()

            change = seasonal_means_future.values - seasonal_means_current.values

            mean_current = seasonal_means_current.values.mean(axis=0)
            mean_future = seasonal_means_future.values.mean(axis=0)

            ##axis0 - time, axis1 -  cell index

            mean_change = change.mean(axis=0)

            #print change[:,mean_change > 200000], seasonal_means_future.values[:,mean_change > 200000], seasonal_means_current.values[:,mean_change > 200000]

            t, pvalue = stats.ttest_1samp(change, 0, axis=0)

            data_map = {
                "current-mean": mean_current,
                "future-mean": mean_future,
                "change": mean_change,
                "p-value": pvalue
            }

            pickle.dump(data_map, open(cache_file, mode="w"))
        else:
            data_map = pickle.load(open(cache_file))
            mean_change = data_map["change"]
            pvalue = data_map["p-value"]
            mean_current = data_map["current-mean"]

        if ax is None:
            plt.figure()

        to_plot = np.ma.masked_all_like(lons2d)

        #mask nonsignificant changes

        #change_arr_significant = np.ma.masked_where(pvalue > 1, mean_change)

        #mean_change[mean_change > levels[-1]] = levels[-1] + 10


        to_plot[x_index, y_index] = mean_change

        print(to_plot.min(), to_plot.max())
        print(pvalue.min(), pvalue.max())

        x, y = b(lons2d, lats2d)

        cmap = cm.get_cmap(name="bwr", lut=len(clevels) - 1)
        #cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/BlueRed.rgb", ncolors=len(levels) - 1)
        #cmap.set_over(cmap(levels[-1]))

        bn = BoundaryNorm(clevels, cmap.N)

        #mask glaciers and oceans
        to_plot = np.ma.masked_where(land_sea_glaciers_mask, to_plot)




        #b.pcolormesh(x, y, to_plot)
        #img = b.pcolormesh(x, y, to_plot, vmin = -100, vmax = 100)

        img = b.pcolormesh(x, y, to_plot, cmap=cmap, vmax=clevels[-1], vmin=clevels[0], norm=bn)
        if ax is None:
            cb = b.colorbar(img, extend="both", ticks=clevels)
            #cb = plt.colorbar(ticks = levels)
            cb.ax.set_title(r"${\rm m^3/s}$")

        b.drawcoastlines(linewidth=0.1)
        b.drawmapboundary(fill_color="0.75")
        #b.drawmeridians(meridians=np.arange(-180, 180, 60))
        #b.drawparallels(circles=np.arange(0, 90, 30))
        b.readshapefile("data/shp/wri_basins2/wribasin", "basin", color="k", linewidth=1)


        if ax is None:
            plt.tight_layout()
            imfile = "offline_rout_{0}_mean_abschange_map_{1}_({2})-({3}).jpeg".format(season, label, p_future, p_current)
            plt.savefig(imfile, dpi=400)

        return img

if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()

    from util import plot_utils

    plot_utils.apply_plot_params(font_size=25)


    clevels = [-3500, -2000, -1500, -750, -500, -100, -50, -1]
    clevels += [-the_lev for the_lev in reversed(clevels)]


    season_months = [
        [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]
    ]

    season_names = [
        "DJF", "MAM", "JJA", "SON"
    ]

    #labels = ["CanESM", "MPI"]
    #paths = [
    #    "/skynet3_rech1/huziy/offline_stfl/canesm/discharge_1958_01_01_00_00.nc",
    #    "/skynet3_rech1/huziy/offline_stfl/mpi/discharge_1958_01_01_00_00.nc"
    #]


    labels = ["MPI", ]
    paths = ["/skynet3_rech1/huziy/offline_stfl/{0}/discharge_1958_01_01_00_00.nc".format(labels[0].lower()), ]

    start_year_current = 1981
    end_year_current = 2010

    start_year_future_list = [2041, 2071]
    end_year_future_list = [2070, 2100]


    for start_year_future, end_year_future in zip(start_year_future_list, end_year_future_list):
        p_current = "{0}-{1}".format(start_year_current, end_year_current)
        p_future = "{0}-{1}".format(start_year_future, end_year_future)



        fig = plt.figure(figsize=(25, 7))
        gs = gridspec.GridSpec(ncols=len(season_names) + 1, nrows=1, width_ratios=len(season_names) * [1.0, ] + [0.05, ])
        col = 0
        img = None
        for smonths, sname in zip(season_months, season_names):
            ax = fig.add_subplot(gs[0, col])
            col += 1
            img = main(months=smonths, season=sname, clevels=clevels, ax = ax, labels = labels, paths = paths)
            ax.set_title(sname + "\n")


        cax = fig.add_subplot(gs[0, col])
        cb = plt.colorbar(img, cax = cax, ticks = clevels, extend="both")
        cb.ax.set_title(r"${\rm m^3/s}$")
        plt.tight_layout()
        imfile = "offline_rout_{0}_mean_abschange_map_{1}_({2})-({3}).jpeg".format("-".join(season_names),
                                                                                   "-".join(labels),
                                                                                   p_future, p_current)
        plt.savefig(imfile, dpi=400)


    print("Hello world")
