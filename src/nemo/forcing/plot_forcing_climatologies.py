from netCDF4 import Dataset
import os

__author__ = 'huziy'

import matplotlib.pyplot as plt


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
            else:
                pass


if __name__ == "__main__":
    main()