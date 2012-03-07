from mpl_toolkits.basemap import Basemap

__author__="huziy"
__date__ ="$Aug 20, 2011 1:45:02 PM$"

import netCDF4 as nc
from rpn.rpn import RPN

import matplotlib.pyplot as plt
import numpy as np

#LAM projection specification
# 1st point - center of the grid
# 2nd point - 90 degrees to the East along the new equator?



def convert(nc_path = 'directions_africa_dx0.44deg.nc'):

    ds = nc.Dataset(nc_path)

    ncNameToRpnName = {'flow_direction_value': 'fldr', 'slope': 'slop', 
                        'channel_length':'leng', 'accumulation_area':'facc',
                        "lake_fraction": "lkfr", "lake_outlet":"lkou"
                      }
    rObj = RPN('infocell.rpn' , mode = 'w')

    #
    ig = []

    #params
    dx = 0.1
    dy = 0.1
    iref = 100
    jref = 100
    xref = 180 #rotated longitude
    yref = 0   #rotated latitude

    #projection parameters
    lon1 = -68.0
    lat1 = 52.0

    lon2 = 16.65
    lat2 = 0.0

    ni = 220
    nj = 220
    x = np.zeros((ni, 1))
    x[:,0] = [xref + (i - iref + 1) * dx for i in xrange(ni)]

    y = np.zeros((1, nj))
    y[0, :] = [yref + (j - jref + 1) * dy for j in xrange(nj)]

    #write coordinates
    rObj.write_2D_field(name="^^", grid_type="E", data=y, typ_var="X", level = 0, ip = range(3),
        lon1=lon1, lat1 = lat1, lon2 = lon2, lat2 = lat2)

    rObj.write_2D_field(name=">>", grid_type="E", data=x, typ_var="X", level = 0, ip = range(3),
            lon1=lon1, lat1 = lat1, lon2 = lon2, lat2 = lat2)

    info = rObj.get_current_info()
    ip_xy = map(lambda x: x.value, info["ip"])
    ig = ip_xy + [0]


    slope_data = None
    flowdir_data = None
    for ncName, rpnName in ncNameToRpnName.iteritems():
        data = ds.variables[ncName][:]
        grid_type = 'Z'
        rObj.write_2D_field(name = rpnName, level = 1, data = data,
            grid_type = grid_type, ig=ig)
        if ncName == "slope":
            slope_data = data
        if ncName == "flow_direction_value":
            flowdir_data = data
    rObj.close()

    ind = (flowdir_data > 0) & (slope_data < 0)
    print flowdir_data[ind], slope_data[ind]
    assert np.all(~ind)


    channel_length = ds.variables['channel_length'][:]
    acc_area = ds.variables['accumulation_area'][:]
    slope = ds.variables['slope'][:]
    fldr = ds.variables['flow_direction_value'][:]

    lons = ds.variables["lon"][:]
    lats = ds.variables["lat"][:]

    basemap = Basemap()
    [x, y] = basemap(lons, lats)






    plt.figure()
    acc_area = np.ma.masked_where( (acc_area < 0), acc_area)
    #basemap.drawcoastlines(linewidth = 0.1)
    #basemap.pcolormesh(x, y, acc_area)
    plt.pcolormesh(acc_area.transpose())

    plt.colorbar()
    plt.title('accumulation area')
    #plt.xlim(x.min(), x.max())
    #plt.ylim(y.min(), y.max())
    plt.savefig("accumulation_area.png")

    plt.figure()
    channel_length = np.ma.masked_where(channel_length < 0, channel_length)
    plt.pcolormesh(channel_length.transpose())
    print "channel_length limits", channel_length.min(), channel_length.max()
    plt.colorbar()
    plt.title('channel length')
    plt.savefig("channel_length.png")

    plt.figure()
    slope = np.ma.masked_where(slope < 0, slope)
    plt.pcolormesh(slope.transpose())
    plt.colorbar()
    plt.title('slope')
    plt.savefig("slope.png")


    plt.figure()
    a2 = 11.0
    a3 = 0.43
    a4 = 1.0
    indx = np.where((slope >= 0) & (channel_length >= 0 ) )
    x = np.zeros(slope.shape)
    x = np.ma.masked_where(slope < 0, x)
    x[indx] = (a2 + a3 * acc_area[indx] ** a4) * channel_length[indx]
    plt.pcolormesh(x.transpose())
    plt.colorbar()

    plt.savefig("bankfull_store.png")
    print ds.variables.keys()


    plt.figure()
    print 'fldr where slope is negative'
    print np.all(fldr[slope < 0] == -1)
    fldr = np.ma.masked_where(fldr < 0, fldr)
    plt.pcolormesh(fldr.transpose())
    plt.colorbar()
    plt.title('fldr')


    print fldr.shape
    fldr = fldr[10:-10, 10:-10]
    channel_length = channel_length[10:-10, 10:-10]
    slope = slope[10:-10, 10:-10]
    print "dir, (98,117): ", fldr[97, 116], fldr[98, 115], channel_length[98,117], slope[98, 117]
    print "acc_a = ", acc_area[97, 116],acc_area[98,115]
    plt.savefig("fldr.png")

    print len(fldr[fldr == 0])

    ds.close()
    pass


import application_properties
if __name__ == "__main__":
    application_properties.set_current_directory()
    convert(nc_path="directions_qc_dx0.1deg.nc")
    print "Hello World"
