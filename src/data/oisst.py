import calendar
from collections import OrderedDict
from pathlib import Path

from rpn.domains import lat_lon
from scipy.spatial import KDTree
from xarray import Dataset
import xarray as xr

import matplotlib.pyplot as plt

cache_dir = Path("oisst_cachedir")

import numpy as np


class OISSTManager(object):

    def __init__(self, thredds_baseurl="https://www.ncei.noaa.gov/thredds/dodsC/OisstBase/NetCDF/AVHRR-AMSR",
                 filename_prefix="amsr-avhrr-v2.", filename_suffix=".nc"):
        """
        :param thredds_baseurl: the data are one day per file in the folders thredds_baseurl/YYYYmm/amsr-avhrr-v2.YYYYMMDD.nc
        there are ice and sst values
        """

        self.base_url = thredds_baseurl
        self._fileurl_pattern = self._build_fileurl_pattern(fname_prefix=filename_prefix, fname_suffix=filename_suffix)



    def _build_fileurl_pattern(self, fname_prefix="", fname_suffix=""):
        """
        :param fname_prefix:
        :param fname_suffix:
        :return: url pattern for accessing file by year, month and day
        """
        return self.base_url + "/{}{:02d}/" + fname_prefix + "{}{:02d}{:02d}" + fname_suffix



    def get_url_for(self, year: int, month: int, day: int):
        """
        Get url of the file corresponding to the depth
        :param year:
        :param month:
        :param day:
        :return:
        """
        return self._fileurl_pattern.format(year, month, year, month, day)




    def download_data_locally(self, start_year=2001, end_year=2011, local_dir="/BIG1/huziy/noaa_oisst_daily"):
        """
        Download the data to a local server
        :param start_year:
        :param end_year:
        :param local_dir:
        """
        for y in range(start_year, end_year + 1):
            # --
            for month in range(1, 13):
                for day in range(1, calendar.monthrange(y, month)[1] + 1):

                    file_url = self.get_url_for(year=y, month=month, day=day)
                    print("Fetching {} ...".format(file_url))

                    try:
                        with xr.open_dataset(file_url) as ds:
                            # assert isinstance(ds, xr.Dataset)
                            local_filepath = Path(local_dir) / Path(file_url).name

                            if local_filepath.exists():
                                print("{} already exists, skipping ...")
                                continue

                            ds.to_netcdf(path=str(local_filepath))
                            print("Saved {} to {}".format(file_url, local_filepath))

                    except OSError as err:
                        print(err)
                        print("Could not find {}".format(file_url))




    def get_seasonal_clim(self, start_year=2002, end_year=2010, season_to_months:dict=None, vname:str= "sst"):

        result = OrderedDict()

        for sname, months in season_to_months.items():

            data_arrays = []

            for y in range(start_year, end_year + 1):
                # --
                for month in months:
                    for day in range(1, calendar.monthrange(y, month)[1] + 1):
                        file_url = self.get_url_for(year=y, month=month, day=day)
                        print("Opening {}".format(file_url))
                        with xr.open_dataset(file_url) as ds:
                            data_arrays.append(ds[vname][0, 0, :, :].load())

            result[sname] = xr.concat(data_arrays, dim="time").mean(dim="time")


        return result



    def get_seasonal_clim_interpolate_to(self, lons=None, lats=None , start_year=2002, end_year=2010, season_to_months:dict=None, vname:str= "sst"):
        """
        Calculate the climatology and then interpolate it to the given lon and lat fields
        :param lons:
        :param lats:
        :param start_year:
        :param end_year:
        :param season_to_months:
        :param vname:
        :return:
        """
        seasclim = self.get_seasonal_clim(start_year=start_year, end_year=end_year, season_to_months=season_to_months, vname=vname)

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())


        inds = None
        seasclim_interpolated = OrderedDict()
        for sname, data in seasclim.items():

            if inds is None:
                lons_s, lats_s = data.coords["lon"][:], data.coords["lat"][:]

                print(data)

                lats_s, lons_s = np.meshgrid(lats_s, lons_s)
                xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())

                ktree = KDTree(list(zip(xs, ys, zs)))

                dists, inds = ktree.query(list(zip(xt, yt, zt)))


            # transpose because the input field's layout is (t,z,lat, lon)
            seasclim_interpolated[sname] = data.values.T.flatten()[inds].reshape(lons.shape)

        return seasclim_interpolated






if __name__ == '__main__':
    sst_obs_manager = OISSTManager()
    sst_obs_manager.download_data_locally()