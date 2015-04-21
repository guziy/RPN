import cartopy
from cartopy import crs
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from nemo.glerl_icecov_data2d_interface import GLERLIceCoverManager
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

    from pathlib import Path

    p = Path(obs_data_path)

    data_files = list(sorted(f for f in p.iterdir()))

    print(data_files[:10])

    gman = GLERLIceCoverManager(data_folder=obs_data_path)

    out_path = "/home/huziy/skynet3_rech1/nemo_obs_for_validation/glerl_icecov.nc"

    start_date = datetime.strptime(data_files[0].name[1:-3], "%Y%m%d")
    with Dataset(out_path, mode="w") as ds:
        ds.createDimension("time")
        ds.createDimension("lon", gman.ncols_target)
        ds.createDimension("lat", gman.ncols_target)

        tvar = ds.createVariable("time", "i4", ("time", ))
        tvar.units = "days since {:%Y-%m-%d %H:%M:%S}".format(start_date)
        tvar.description = "ice cover data from GLERL"
        dvar = ds.createVariable("ice_cover", "f4", ("time", "lon", "lat"))


        lon_var = ds.createVariable("lon", "f4", ("lon", "lat"))
        lat_var = ds.createVariable("lat", "f4", ("lon", "lat"))

        # save the coordinates
        lon_var[:] = gman.lons2d_target
        lat_var[:] = gman.lats2d_target


        import matplotlib.pyplot as plt

        for i, fpath in enumerate(data_files):

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