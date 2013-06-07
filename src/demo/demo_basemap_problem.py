from mpl_toolkits.basemap import Basemap
__author__ = 'huziy'

import numpy as np
import netCDF4 as nc

def main():
    ds = nc.Dataset("../../grid_arctic.nc")
    data = ds.variables["tas"][:].squeeze()
    lon2d_nc = ds.variables["lon"][:]
    lat2d_nc = ds.variables["lat"][:]


    rplon, rplat = -115.850877297, 21.1216893478
    lon_0 = -123.403723307


    basemap = Basemap(projection="rotpole", o_lon_p=rplon, o_lat_p=rplat,
                    lon_0 = lon_0 - 180,
                    llcrnrlon=lon2d_nc[0,0], llcrnrlat=lat2d_nc[0,0],
                    urcrnrlon=lon2d_nc[-1,-1], urcrnrlat=lat2d_nc[-1, -1],
                    resolution="l"
    )

    import matplotlib.pyplot as plt
    x, y = basemap(lon2d_nc, lat2d_nc)
    basemap.contourf( x, y, data)
    basemap.colorbar()
    basemap.drawcoastlines()
    basemap.drawmeridians(np.arange(-180, 180, 50))
    basemap.drawparallels(np.arange(-90, 90, 30))
    plt.show()

    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  