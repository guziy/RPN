from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from domains.rotated_lat_lon import RotatedLatLon

__author__ = 'huziy'


from netCDF4 import Dataset
import matplotlib.pyplot as plt
from domains import grid_config
import numpy as np

from rpn.rpn import RPN


def plot_acc_area_with_glaciers():
    gmask_vname = "VF"
    gmask_level = 2
    gmask_path = "/skynet1_exec2/winger/Glacier/Offline/Input/geophys_West_NA_0.25deg_144x115.fst"
    r = RPN(gmask_path)

    gmask = r.get_first_record_for_name_and_level(varname=gmask_vname,
                                                  level=gmask_level)

    proj_params = r.get_proj_parameters_for_the_last_read_rec()
    rll = RotatedLatLon(**proj_params)
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)
    gmask = np.ma.masked_where(gmask < 0.01, gmask)
    mask_value = 0.25
    gmask[~gmask.mask] = mask_value


    path = "/b2_fs2/huziy/directions_north_america_0.25deg_glaciers.nc"
    ds = Dataset(path)

    var_name = "accumulation_area"
    data = ds.variables[var_name][:]

    data = np.ma.masked_where(data <= 0, data)



    x, y = basemap(lons, lats)
    im = basemap.pcolormesh(x, y, data, norm = LogNorm(vmin=1e3, vmax=1e7), cmap = cm.get_cmap("jet", 12))
    cmap = cm.get_cmap("gray_r", 10)
    basemap.pcolormesh(x, y, gmask, cmap = cmap, vmin = 0., vmax = 1.)
    basemap.drawcoastlines(linewidth=0.5)
    cb = basemap.colorbar(im)
    cb.ax.tick_params(labelsize = 25)

    plt.legend([Rectangle((0, 0), 5, 5, fc = cmap(mask_value)), ], ["Glaciers", ], loc = 3)
    plt.show()



def main():
    path = "/b2_fs2/huziy/directions_north_america_0.1375deg.nc"
    ds = Dataset(path)

    var_name = "accumulation_area"
    data = ds.variables[var_name][:]

    data = np.ma.masked_where(data <= 0, data)

    print(list(ds.variables.keys()))

    lons = ds.variables["lon"][:]
    lats = ds.variables["lat"][:]

    rll = grid_config.get_rotpole_for_na_glaciers()

    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)

    x, y = basemap(lons, lats)
    im = basemap.pcolormesh(x, y, data, norm = LogNorm())
    basemap.drawcoastlines(linewidth=0.5)
    basemap.colorbar(im)
    plt.show()


if __name__ == "__main__":
    #main()
    plot_acc_area_with_glaciers()