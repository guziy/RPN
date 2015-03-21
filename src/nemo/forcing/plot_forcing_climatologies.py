from netCDF4 import Dataset
import os
from datetime import timedelta, datetime

__author__ = 'huziy'

import matplotlib.pyplot as plt
import pandas as pd

def plot_wind(basemap, x, y, u, v, ax = None):
    pass


def plot_scalar_field(basemap, x, y, title_to_field, ax = None):
    pass


def main(path = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/DFS5.2_clim"):
    """
    Considers that the path contains files with calculated climatological means, with frequency < 1 year,
    i.e. daily, monthly...
    plots the first 3d variable found in a netcdf file
    :param path:
    :param frequency: time step between data fields
    """

    name_to_data = {}

    for f in os.listdir(path):
        f_path = os.path.join(path, f)
        ds = Dataset(f_path)

        for vname, var in ds.variables.iteritems():


            if vname not in ["u10", "v10"]:
                if var.ndim == 3:
                    data = var[:]

                    ntimes = data.shape[0] - 1
                    dtsec = (365 * 24 * 60.0 * 60.0) / float(ntimes)

                    dt = timedelta(seconds = dtsec)
                    start_date = datetime(2001, 1, 1)
                    end_date = start_date + ntimes * dt
                    dates = pd.DatetimeIndex(start = start_date, end = end_date,
                                             freq = pd.DateOffset(seconds = dtsec))
                    print vname
                    print dt, start_date, end_date
                    print dates[0], dates[-1]

                    ny, nx = data[0, :, :].shape
                    p = pd.Panel(data=data, items=dates, major_axis=range(ny), minor_axis=range(nx))



            else:
                pass


if __name__ == "__main__":
    main()