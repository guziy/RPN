from netCDF4 import Dataset
from datetime import datetime
import pickle
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
import os
import pandas
from domains.rotated_lat_lon import RotatedLatLon

__author__ = 'huziy'

import numpy as np


def get_arctic_basemap(lons2d, lats2d, lon1 = 60, lat1 = 90, lon2 = -30, lat2 = 0, resolution = "l"):
    rll = RotatedLatLon(lon1 = lon1, lat1 = lat1, lon2 = lon2, lat2 = lat2)
    rplon, rplat = rll.get_north_pole_coords()
    lon_0, lat_0 = rll.get_true_pole_coords_in_rotated_system()

    basemap = Basemap(projection="rotpole", o_lon_p=rplon, o_lat_p=rplat,
             lon_0 = lon_0 - 180,
             llcrnrlon=lons2d[0,0], llcrnrlat=lats2d[0,0],
             urcrnrlon=lons2d[-1,-1], urcrnrlat=lats2d[-1, -1],
             resolution=resolution, round= True
    )
    return basemap



TIME_FORMAT = "%Y_%m_%d_%H_%M"

def main(months= None, season="DJF"):
    if not months: months = [12, 1, 2]
    import matplotlib.pyplot as plt
    labels = ["CanESM", "MPI"]
    paths = [
             "/skynet3_rech1/huziy/offline_stfl/canesm/discharge_1958_01_01_00_00.nc",
             "/skynet3_rech1/huziy/offline_stfl/mpi/discharge_1958_01_01_00_00.nc"
             ]


    start_year = 2071
    end_year = 2100

    lons2d = None
    lats2d = None


    x_index = None
    y_index = None
    mean_data = None


    b = None



    for the_path, label in zip( paths, labels ):
        ds = Dataset(the_path)
        if lons2d is None:
            lons2d = ds.variables["longitude"][:]
            lats2d = ds.variables["latitude"][:]

            b = get_arctic_basemap(lons2d, lats2d)


            x_index = ds.variables["x_index"][:]
            y_index = ds.variables["y_index"][:]

        cache_file = "_".join([str(m) for m in months]) + "_{0}_{1}_{2}_mean_cache.bin".format(start_year, end_year, label)
        if not os.path.isfile(cache_file):
            time_str = ds.variables["time"][:]
            times = [ datetime.strptime("".join(t_s), TIME_FORMAT) for t_s in time_str]
            data = ds.variables["water_discharge_accumulated"][:]
            df = pandas.DataFrame(data=data, index=times)
            df["year"] = df.index.map(lambda d: d.year)
            df["month"] = df.index.map(lambda d: d.month)

            print df.shape
            mean_data = df.ix[ (df.year >= start_year) & (df.year <= end_year) & df.month.isin(months), :].mean(axis = 0)
            mean_data = mean_data.drop(["year", "month"]) #no need of month and year
            pickle.dump(mean_data, open(cache_file, mode="w"))
        else:
            mean_data = pickle.load(open(cache_file))

        plt.figure()
        to_plot = np.ma.masked_all_like(lons2d)
        print x_index.shape
        print mean_data.values.shape

        to_plot[x_index, y_index] = mean_data.values

        print to_plot.min(), to_plot.max()

        x, y = b(lons2d, lats2d)

        levels = [0, 100, 200, 500, 1000, 5000, 10000, 15000, 20000, 30000]
        cmap = cm.get_cmap(name="jet", lut = len(levels) - 1)

        bn = BoundaryNorm(levels, cmap.N)

        b.contourf(x, y, to_plot, levels = levels, norm = bn, cmap = cmap)
        b.colorbar()
        b.drawcoastlines(linewidth=0.1)

        b.drawmeridians(meridians=np.arange(-180, 180, 60))
        b.drawparallels(circles=np.arange(0,90, 30))

        b.readshapefile("data/shp/wri_basins2/wribasin", "basin", color = "0.75", linewidth=2)
        plt.tight_layout()
        plt.savefig("offline_rout_{0}_mean_map_{1}.pdf".format(season, label))


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=20, height_cm=20, font_size=26)

    season_months = [
        [12,1,2], [3,4,5], [6,7,8], [9,10,11]
    ]

    season_names = [
        "DJF", "MAM", "JJA", "SON"
    ]


    for smonths, sname in zip(season_months, season_names):
        main(months= smonths, season=sname)
    print "Hello world"
  