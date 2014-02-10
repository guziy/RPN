__author__ = 'huziy'

import pandas as pd
from netCDF4 import Dataset, Variable
from netCDF4 import num2date
import matplotlib.pyplot as plt
import numpy as np


def main(var_name = "sst"):
    inpath = "/skynet3_rech1/huziy/NEMO/WORK_GRTLKS/data_from_NOAA/sst.wkmean.1981-1989.nc"
    outpath = "/skynet3_rech1/huziy/NEMO/WORK_GRTLKS/data_from_NOAA/sst.monclim.1981-1989.nc"

    ds = Dataset(inpath)

    in_var = ds.variables[var_name]
    data = in_var[:]
    time_var = ds.variables["time"]
    time_objs = num2date(time_var[:], time_var.units)

    ny, nx = data.shape[1:]

    lonVar = ds.variables["lon"]
    latVar = ds.variables["lat"]

    assert isinstance(lonVar, Variable)
    print lonVar.ndim



    panel = pd.Panel(data=data, items=time_objs, major_axis=range(ny), minor_axis=range(nx))
    mean_clim = panel.groupby(lambda d: d.month, axis="items").mean()

    data_out = [
        mean_clim[m].values for m in range(1, 13)
    ]
    data_out = np.asarray(data_out)

    ds_out = Dataset(outpath, "w", format="NETCDF3_CLASSIC")

    ds_out.createDimension("y", ny)
    ds_out.createDimension("x", nx)
    ds_out.createDimension("time", None)


    out_var = ds_out.createVariable(var_name, "f4", ("time", "y", "x"))
    out_var.units = in_var.units
    out_var.long_name = in_var.long_name


    lon_out_var = ds_out.createVariable("lon", "f4", ("x", ))
    lat_out_var = ds_out.createVariable("lat", "f4", ("y", ))

    for att_name in lonVar.ncattrs():
        assert isinstance(lon_out_var, Variable)
        lon_out_var.setncattr(att_name, lonVar.getncattr(att_name))
        lat_out_var.setncattr(att_name, latVar.getncattr(att_name))

    lon_out_var[:] = lonVar[:]
    lat_out_var[:] = latVar[:]
    out_var[:] = data_out


    ds_out.close()

    pass

if __name__ == "__main__":
    main()