from matplotlib.colors import LogNorm

__author__ = 'huziy'


from netCDF4 import Dataset
import matplotlib.pyplot as plt
from domains import grid_config
import numpy as np


def main():
    path = "/b2_fs2/huziy/directions_north_america_0.1375deg.nc"
    ds = Dataset(path)

    var_name = "accumulation_area"
    data = ds.variables[var_name][:]

    data = np.ma.masked_where(data <= 0, data)

    print ds.variables.keys()

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
    main()