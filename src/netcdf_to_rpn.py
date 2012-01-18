from mpl_toolkits.basemap import Basemap

__author__="huziy"
__date__ ="$Aug 20, 2011 1:45:02 PM$"

import netCDF4 as nc
from rpn import RPN

import matplotlib.pyplot as plt
import numpy as np

def convert(nc_path = 'directions_africa_dx0.44deg.nc'):
    ds = nc.Dataset(nc_path)

    ncNameToRpnName = {'flow_direction_value': 'fldr', 'slope': 'slop', 
                        'channel_length':'leng', 'accumulation_area':'facc',
                        'lon': '>>', 'lat' : '^^'
                      }
    rObj = RPN('infocell.rpn' , mode = 'w')
    for ncName, rpnName in ncNameToRpnName.iteritems():
        data = ds.variables[ncName][:]
        print ncName, data.min(), data.max(), data.mean()
        grid_type = 'E' if rpnName in ['>>', '^^'] else 'Z'
        rObj.write_2D_field(name = rpnName, level = 1, data = data, grid_type = grid_type)
    rObj.close()



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
    convert()
    print "Hello World"
