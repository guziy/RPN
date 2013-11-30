from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

#path = '/home/huziy/skynet1_rech3/Converters/NetCDF_converter/wm201_Arctic_JJA_1990-2008_moyenneDesMoyennes.nc'
path = '/home/huziy/skynet1_rech3/Converters/NetCDF_converter/pm1989010100_00000498p_0.44deg_africa_PR.nc'


def _get_data(nc):
    lats = nc.variables['lat'][:, :]
    lons = nc.variables['lon'][:, :]

    print nc.variables.keys()
    if "PR" in nc.variables:
        data = nc.variables["PR"][0].squeeze()
    elif "air" in nc.variables:
        data = nc.variables["air"][0].squeeze()
    elif "preacc" in nc.variables:
        data = nc.variables['preacc'][0].squeeze()
    return data, lons, lats

def demo1():
    nc = Dataset(path)
    data, lons, lats = _get_data(nc)

    nc.close()
    data = np.ma.masked_values(data, -999.)
    m = Basemap(projection='npstere', lon_0=10, boundinglat=30, resolution='l')
    x, y = m(lons, lats)
    print data.min(), data.max()
    m.drawcoastlines()
    m.contourf(x, y, data, 20)
    m.drawmeridians(np.arange(-180, 180, 20))
    m.drawparallels(np.arange(20, 80, 20))

    plt.title('rotated pole data')


def demo2():
    nc = Dataset(path)
    data, lons, lats = _get_data(nc)
    data = np.ma.masked_values(data, -999.)

    nc.close()
    nx, ny = lons.shape

    plt.figure()
    #    basemap = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(lons2d=lons, lats2d= lats,
    #        lon_1=60, lat_1 = 90, lon_2=-30, lat_2=0.
    #    )


    prj_info = None

    lonp = 0
    latp = 90.0
    lon_0 = -180
    if nc.variables.has_key("rotated_pole"):
        prj_info = nc.variables["rotated_pole"]
        print dir(prj_info)
        lonp = prj_info.grid_north_pole_longitude
        latp = prj_info.grid_north_pole_latitude
        lon_0 = prj_info.north_pole_grid_longitude

    basemap = Basemap(projection="rotpole", o_lon_p=lonp, o_lat_p=latp, lon_0=lon_0 - 180,
                      llcrnrlon=lons[0, 0], llcrnrlat=lats[0, 0],
                      urcrnrlon=lons[-1, -1], urcrnrlat=lats[-1, -1])

    x, y = basemap(lons, lats)

    basemap.drawmeridians(np.arange(-180, 180, 20))
    basemap.drawparallels(np.arange(-90, 90, 30))

    basemap.contourf(x, y, data)
    basemap.drawcoastlines()
    basemap.colorbar()


def main():
    #demo1()
    demo2()
    plt.show()
    pass


if __name__ == "__main__":
    main()
    print "Hello world"
  