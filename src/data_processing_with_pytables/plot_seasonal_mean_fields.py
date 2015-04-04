import os
from pandas.core.daterange import DateRange
from pandas.tseries.index import DatetimeIndex
from pandas.tseries.offsets import DateOffset

__author__ = 'huziy'

import numpy as np
import tables as tb


from netCDF4 import num2date, Dataset


def main():
    nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

    sim_name = "crcm5-r"

    sim_folder = os.path.join(nc_db_folder, "{0}".format(sim_name))

    var_name = "TT"

    nc_path = os.path.join(sim_folder, "{0}_all.nc4".format(var_name))

    ds = Dataset(nc_path)

    timeVar = ds.variables["time"]
    print(timeVar.units)


    t0 = num2date(timeVar[0], timeVar.units)
    t1 = num2date(timeVar[1], timeVar.units)
    tf = num2date(timeVar[-1], timeVar.units)

    dt = t1 - t0
    dr = DatetimeIndex(start = t0, end = tf, freq = DateOffset(seconds = dt.seconds))

    #print len( dr.tolist() )
    print(t0, t1, tf)
    #print help( h.root.TT )








    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  