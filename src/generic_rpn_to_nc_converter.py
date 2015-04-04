from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap
from rpn.rpn import RPN
import netCDF4 as nc
__author__ = 'huziy'

import numpy as np


#all fields in the input file should have the same x,y coordinates

def diagnose_mean():
    path = "/home/huziy/skynet1_rech3/cordex/for_Samira/Africa_0.44deg_ERA40-Int1.5_E21981-2010/dailyAfrica_0.44deg_ERA40-Int1.5_E21981-2010TRAF.nc"

    ds = nc.Dataset(path)
    traf = ds.variables["TRAF"][:,0,:,:]

    traf_m = np.mean(traf, axis = 0) * 24 *60 * 60 * 365 #transform to mm/day
    lon = ds.variables["longitude"][:]
    lat = ds.variables["latitude"][:]


    lon[lon > 180] -= 360


    levels = [0,0.1,1,5,10,25, 50,100, 200, 300, 600, 1000,1500,2000,2500,3000,5000]
    cMap = get_cmap("jet", len(levels) - 1 )
    bn = BoundaryNorm(levels, cMap.N)

    ll_lon, ur_lon = np.min(lon), np.max(lon)

    ll_lat, ur_lat = np.min(lat), np.max(lat)
    traf_min = 0.1
    traf_m = np.ma.masked_where(traf_m < traf_min, traf_m)

    import matplotlib.pyplot as plt
    b = Basemap(projection="merc",llcrnrlon=ll_lon, llcrnrlat=ll_lat, urcrnrlon=ur_lon, urcrnrlat=ur_lat)
    x,y = b(lon, lat)
    plt.figure()
    b.pcolormesh(x, y, traf_m, norm = bn, cmap = cMap)
    cb = plt.colorbar()
    cb.ax.set_title("mm/year")
    b.drawcoastlines()
    plt.show()


def main():
    rpn_path = "/home/huziy/skynet1_rech3/cordex/for_Samira/Africa_0.44deg_ERA40-Int1.5_E21981-2010/dailyAfrica_0.44deg_ERA40-Int1.5_E21981-2010TRAF"
    nc_path = rpn_path + ".nc"

    varname = "TRAF"
    time_units = "days since 1981-01-01 00:00:00"


    rObj = RPN(rpn_path)
    lons2d, lats2d = rObj.get_longitudes_and_latitudes()
    rObj.suppress_log_messages()
    data = rObj.get_4d_field(name=varname)

    rObj.close()


    ds = nc.Dataset(nc_path, "w", format="NETCDF3_CLASSIC")

    nx, ny = lons2d.shape

    levels = list(data.items())[0][1].keys()

    ds.createDimension("lon", nx)
    ds.createDimension("lat", ny)
    ds.createDimension("level", len(levels))
    ds.createDimension("time", None)

    var = ds.createVariable(varname, "f4", dimensions=("time","level", "lon", "lat"))
    lonVar = ds.createVariable("longitude", "f4", dimensions=("lon", "lat"))
    latVar = ds.createVariable("latitude", "f4", dimensions=("lon", "lat"))
    timeVar = ds.createVariable("time", "f4", dimensions=("time",))


    times_sorted = list( sorted( data.keys() ) )
    levels_sorted = list( sorted(levels) )

    data_arr = [
        [ data[t][lev] for lev in levels_sorted ] for t in times_sorted
    ]

    data_arr = np.array(data_arr)
    var[:] = data_arr

    timeVar.units = time_units
    times_num = nc.date2num(times_sorted, time_units)
    timeVar[:] = times_num

    lonVar[:] = lons2d
    latVar[:] = lats2d
    ds.close()


    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    #main()
    diagnose_mean()
    print("Hello world")
  