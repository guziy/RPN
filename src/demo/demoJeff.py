__author__ = 'huziy'

import numpy as np
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

path = '/home/huziy/skynet1_rech3/Converters/NetCDF_converter/wm201_Arctic_JJA_1990-2008_moyenneDesMoyennes.nc'
def demo1():
    nc = Dataset(path)
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]
    data = nc.variables['air'][0].squeeze()
    nc.close()
    data = np.ma.masked_values(data,-999.)
    m = Basemap(projection='npstere',lon_0=10,boundinglat=30,resolution='l')
    x,y = m(lons,lats)
    print data.min(), data.max()
    m.drawcoastlines()
    m.contourf(x,y,data,20)
    m.drawmeridians(np.arange(-180,180,20))
    m.drawparallels(np.arange(20,80,20))

    plt.title('rotated pole data')

def demo2():

    nc = Dataset(path)
    lats = nc.variables['lat'][:, :]
    lons = nc.variables['lon'][:, :]
    data = nc.variables['air'][0,0,:,:]
    data = np.ma.masked_values(data,-999.)
    nc.close()
    nx, ny = lons.shape

    lon_0, lat_0 = lons[nx//2, ny//2], lats[nx//2, ny//2]

    plt.figure()
    basemap = Basemap(projection = "omerc", lon_1=60, lat_1 = 89.9, lon_2=-30, lat_2=0.1, no_rot=True,
        lon_0 = lon_0, lat_0 = lat_0,
        llcrnrlon=lons[0, 0], llcrnrlat=lats[0,0],
        urcrnrlon=lons[-1, -1], urcrnrlat=lats[-1, -1]

    )

    x, y = basemap(lons, lats)


    basemap.contourf(x, y, data)
    basemap.drawcoastlines()
    basemap.colorbar()



def main():
    demo1()
    demo2()
    plt.show()
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  