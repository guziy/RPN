from memory_profiler import profile
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from nemo.glerl_icecov_data2d_interface import GLERLIceCoverManager, get_date_from_nic_cis_filepath
import numpy as np
from datetime import datetime

__author__ = 'huziy'


class GlerlGrid(object):
    known_grids = {}

    def __init__(self, grid_descriptor):
        """
        Holds longitudes and latitudes (2d) of the GLERL grids. There are 2 possibilities for grid descriptors:
            -- 1024x1024
            -- 516x510
        """
        self._descriptor = grid_descriptor
        self._lons2d = None
        self._lats2d = None

    @property
    def lons2d(self):
        if self._lons2d is None:
            pass
        return self._lons2d

    @property
    def lats2d(self):
        if self._lats2d is None:
            pass
        return self._lats2d

    def __getitem__(self, grid_descriptor):
        if grid_descriptor in self.known_grids:
            return self.known_grids[grid_descriptor]



def main():
    obs_data_path = "/home/huziy/skynet3_rech1/nemo_obs_for_validation/glerl_icecover_all_files"
    obs_data_path_1973_2002 = "/HOME/huziy/skynet3_rech1/nemo_obs_for_validation/ice_cover_glk/daily_grids_1973_2002/data"


    from pathlib import Path

    p = Path(obs_data_path)

    data_files = list(sorted(f for f in p.iterdir()))

    print(data_files[:10])

    gman = GLERLIceCoverManager(data_folder=obs_data_path)

    out_path = "/home/huziy/skynet3_rech1/nemo_obs_for_validation/glerl_icecov1_fix.nc"

    # start_date = datetime.strptime(data_files[0].name[1:-3], "%Y%m%d")
    start_date = datetime(1972, 1, 1)
    with Dataset(out_path, mode="w") as ds:
        ds.createDimension("time")
        ds.createDimension("x", gman.ncols_target)
        ds.createDimension("y", gman.ncols_target)

        tvar = ds.createVariable("time", "i4", ("time", ))
        tvar.units = "days since {:%Y-%m-%d %H:%M:%S}".format(start_date)
        tvar.description = "ice cover data from GLERL"
        assert isinstance(ds, Dataset)
        dvar = ds.createVariable("ice_cover", "f4", ("time", "x", "y"), zlib=True, least_significant_digit=3)
        dvar.coordinates = "lon lat"
        dvar.missing_value = 1e20


        lon_var = ds.createVariable("lon", "f4", ("x", "y"))
        lat_var = ds.createVariable("lat", "f4", ("x", "y"))

        # save the coordinates
        lon_var[:] = gman.lons2d_target
        lat_var[:] = gman.lats2d_target


        i1 = 0
        # write the data for 1973-2002 period
        for i, fpath in enumerate(sorted(Path(obs_data_path_1973_2002).iterdir(), key=lambda zp: get_date_from_nic_cis_filepath(zp))):

            if not fpath.name.lower()[-3:] in ["cis", "nic"]:
                continue

            the_date = get_date_from_nic_cis_filepath(fpath)


            # Avoid duplicates
            if the_date.year >= 2003:
                continue

            if the_date.year == 2002 and the_date.month >= 12:
                continue


            data = gman.get_data_from_file_interpolate_if_needed(fpath)

            dvar[i, :, :] = data

            dt = the_date - start_date

            tvar[i] = dt.total_seconds() / (3600.0 * 24.0)


            # debug for testing
            # if fpath.name[-5] != "0":
            #     import matplotlib.pyplot as plt
            #
            #     plt.figure()
            #     prj = crs.PlateCarree()
            #     ax = plt.axes(projection=prj)
            #     ax.pcolormesh(gman.lons2d_target, gman.lats2d_target, data, transform=prj)
            #     ax.add_feature(cartopy.feature.LAKES, facecolor="none", edgecolor="k", linewidth=2)
            #     ax.coastlines()
            #     plt.show()
            #
            #     if True:
            #         raise Exception()

            i1 += 1


        print("processed data for 1973-2002 period")

        for i, fpath in enumerate(data_files, start=i1):

            current_date = datetime.strptime(fpath.name[1:-3], "%Y%m%d")

            print(str(fpath))
            print(current_date)
            data = gman.get_data_from_file_interpolate_if_needed(fpath)
            print(data.shape)
            dvar[i, :, :] = data

            # plt.figure()
            # prj = crs.PlateCarree()
            # ax = plt.axes(projection=prj)
            # ax.pcolormesh(gman.lons2d_target, gman.lats2d_target, data, transform=prj)
            # ax.add_feature(cartopy.feature.LAKES, facecolor="none", edgecolor="k", linewidth=2)
            # ax.coastlines()
            # plt.show()

            dt = current_date - start_date

            tvar[i] = dt.total_seconds() / (3600.0 * 24.0)



if __name__ == '__main__':
    main()
