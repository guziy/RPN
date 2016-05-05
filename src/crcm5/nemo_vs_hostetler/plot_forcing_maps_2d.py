from application_properties import main_decorator

import matplotlib.pyplot as plt

from netCDF4 import Dataset


@main_decorator
def main():

    path_to_erai = "~/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/snow.nc"
    bathy_file = "~/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/bathy_meter.nc"

    with Dataset(bathy_file) as ds:
        bath = ds.variables["Bathymetry"][:]
        lons, lats = ds.variables["nav_lon"][:], ds.variables["nav_lat"][:]








if __name__ == '__main__':
    main()