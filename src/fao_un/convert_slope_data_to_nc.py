import os
from netCDF4 import Dataset
import numpy as np
__author__ = 'huziy'

import pandas as pd




def main():
    folder_path = "/home/huziy/skynet3_rech1/Global_terrain_slopes_30s"
    out_filename = "slopes_30s.nc"
    in_fname_pattern = "GloSlopesCl{0}_30as.asc"



    nclasses = 1
    params = {}

    out_path = os.path.join(folder_path, out_filename)
    ds = Dataset(out_path, "w")

    for cl in range(1, nclasses + 1):
        print "cl = {0}".format(cl)
        inpath = os.path.join(folder_path, in_fname_pattern.format(cl))
        if cl == 1:  # generate lon/lat
            with open(inpath) as f:
                for i in range(6):
                    key, val = f.readline().strip().split()
                    val = float(val)
                    params[key] = val

                print params
                # params["nrows"] //= 1000
                # params["ncols"] //= 1000
                params["nrows"] = int(np.round(params["nrows"]))
                params["ncols"] = int(np.round(params["ncols"]))
                print params


                d = params["cellsize"]
                lon1d = [params["xllcorner"] + i * d for i in range(params["ncols"])]
                lat1d = [params["yllcorner"] + i * d for i in range(params["nrows"])]
                lon2d, lat2d = np.meshgrid(lon1d, lat1d)
                ds.createDimension("x", len(lon1d))
                ds.createDimension("y", len(lat1d))

                lon_var = ds.createVariable("lon", "f4", ("y", "x"))
                lat_var = ds.createVariable("lat", "f4", ("y", "x"))

                lon_var[:] = lon2d
                lat_var[:] = lat2d

        data_var = ds.createVariable("class{0}".format(cl), np.uint8, ("y", "x"), chunksizes=(100, 100))
        data_var.missing_value = params["NODATA_value"]
        data = np.loadtxt(inpath, skiprows=6, dtype=np.uint8)
        data = np.flipud(data)
        print data.min(), data.max()

        data_var[:] = data



    ds.close()



if __name__ == "__main__":
    main()