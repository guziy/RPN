


# Read monthly SWE (mm) data from binary files on the EASE grid
# calculate climatologies and interpolate
# Binary, 721x721 16-bit signed, little-endian (LSB) integers, by row
#
# Each monthly binary data file with the file extension ".NSIDC8" contains a flat, binary array of 16-bit signed,
# little-endian (LSB) integers, 721 columns by 721 rows (row-major order, i.e. the top row of the array comprises the
# first 721 values in the file, etc.).
#

from collections import OrderedDict

import numpy as np
import os
import struct

from mpl_toolkits.basemap import Basemap
from pyresample import geometry, kd_tree

import calendar

import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from util.geo import lat_lon

NBYTES_PER_FIELD_SWE = 2
NBYTES_PER_FIELD_LON_LAT = 4


# 25 km, Northern hemisphere longitudes and latitludes files
LONS_PATH = "/RESCUE/skynet3_rech1/huziy/obs_data/SWE/NSIDC-EASE-grid-monthly/nsidc0271v01/latlon/low_res/NLLONLSB"
LATS_PATH = "/RESCUE/skynet3_rech1/huziy/obs_data/SWE/NSIDC-EASE-grid-monthly/nsidc0271v01/latlon/low_res/NLLATLSB"

# Needed to convert from the binary format stored in files
LONLAT_CONVERSION_COEF = 100000.0



def _get_array_from_file(path, nbytes_per_field=NBYTES_PER_FIELD_SWE, dtype="h", mask_negative=False):
    data_bin = open(path, mode="rb").read()
    ncols = (len(data_bin) // nbytes_per_field) ** 0.5
    ncols = int(ncols + 0.5)
    nrows = ncols

    v = struct.unpack("<{}{}".format(ncols * nrows, dtype), data_bin)

    arr = np.array(v)
    arr = arr.reshape(nrows, ncols)
    arr = np.flipud(arr).astype(float)

    if mask_negative:
        arr = np.ma.masked_where(arr < 0, arr)
        # arr[arr < 0] = np.nan

    return arr


def _get_year_and_month(filename):
    part = filename.split(".", maxsplit=1)[0][2:]
    return int(part[:-2]), int(part[-2:])



class EaseSweManager(object):

    def __init__(self, data_folder="/RESCUE/skynet3_rech1/huziy/obs_data/SWE/NSIDC-EASE-grid-monthly/nsidc0271v01/north/all",
                 numdays_folder="/HOME/huziy/skynet3_rech1/obs_data/SWE/NSIDC-EASE-grid-monthly/nsidc0271v01/north/numdays"):

        self.data_folder = data_folder
        self.numdays_folder = numdays_folder

        self.lons = _get_array_from_file(LONS_PATH, nbytes_per_field=NBYTES_PER_FIELD_LON_LAT, dtype="i") / LONLAT_CONVERSION_COEF
        self.lats = _get_array_from_file(LATS_PATH, nbytes_per_field=NBYTES_PER_FIELD_LON_LAT, dtype="i") / LONLAT_CONVERSION_COEF
        self.kdtree = None


    def get_seasonal_clim_interpolated_to(self, target_lon2d=None, target_lat2d=None, season_to_months=None, start_year=-np.Inf, end_year=np.Inf):
        season_to_clim = self.get_seasonal_clim(season_to_months=season_to_months, start_year=start_year, end_year=end_year)

        selection = (self.lons <= 180) & (self.lons >= -180) & (self.lats >= -90) & (self.lats <= 90)

        if self.kdtree is None:
            x, y, z = lat_lon.lon_lat_to_cartesian(self.lons[selection], self.lats[selection])
            self.kdtree = KDTree(list(zip(x, y, z)))

        x1, y1, z1 = lat_lon.lon_lat_to_cartesian(target_lon2d.flatten(), target_lat2d.flatten())
        dists, inds = self.kdtree.query(list(zip(x1, y1, z1)))
        season_to_clim_interpolated = OrderedDict()

        for season, clim in season_to_clim.items():
            season_to_clim_interpolated[season] = clim[selection][inds].reshape(target_lon2d.shape)

        #
        plt.show()


        return season_to_clim_interpolated


    def get_seasonal_clim(self, season_to_months=None, start_year=-np.Inf, end_year=np.Inf):

        """
        :param season_to_months:
        :param start_year:
        :param end_year:
        :return: array(latdim, londim)
        """
        season_to_clim_field = OrderedDict()


        # select files for each season
        season_to_file_list = OrderedDict([(s, []) for s in season_to_months])
        season_to_nday_list = OrderedDict([(s, []) for s in season_to_months])
        for fn in os.listdir(self.data_folder):
            y, m = _get_year_and_month(fn)


            if y < start_year or y > end_year:
                continue


            for season, months in season_to_months.items():
                if m in months:
                    season_to_file_list[season].append(os.path.join(self.data_folder, fn))
                    season_to_nday_list[season].append(os.path.join(self.numdays_folder, "{}.num".format(fn)))
                    break



        # Calculate seasonal means
        for season, flist in season_to_file_list.items():

            all_for_season = np.ma.array([_get_array_from_file(os.path.join(self.data_folder, fp), mask_negative=True) for fp in flist])
            weights = np.ma.asarray([_get_array_from_file(os.path.join(self.data_folder, fp), mask_negative=True) for fp in season_to_nday_list[season]]).astype(float)
            weights /= weights.sum(axis=0)[np.newaxis, :, :]


            print(weights.sum(), len(weights))

            print(weights.min(), weights.max())
            season_to_clim_field[season] = np.ma.sum(all_for_season * weights, axis=0)


        return season_to_clim_field


def test():
    manager = EaseSweManager()

    season_to_months = OrderedDict([
        ("Winter", [12, 1, 2]),
         ("Spring", [3, 4, 5])
    ])

    manager.get_seasonal_clim(season_to_months=season_to_months, start_year=1980, end_year=2006)



if __name__ == '__main__':
    test()
