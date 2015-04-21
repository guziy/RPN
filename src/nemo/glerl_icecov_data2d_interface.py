from datetime import datetime
from mpl_toolkits.basemap import Basemap
from scipy.spatial.ckdtree import cKDTree
from util.geo import lat_lon

__author__ = 'huziy'

import os
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


def update_grid_info(func):
    """
    decorates GLERLIceCoverManager.get_icecover_interpolated_to in order to always have updated grid information
    :param func:
    :return:
    """

    def update(*args, **kwargs):
        the_date = kwargs["the_date"]
        # args[0] == self
        args[0].read_location_info(GLERLIceCoverManager.date_to_fname(the_date))
        return func(*args, **kwargs)

    return update


class GLERLIceCoverManager(object):
    # Projection params for the (1024x1024 grid)
    slat = 38.8744  # Southern latitude
    wlon = -92.4106  # Western longitude
    elon = -75.8690  # Eastern longitude
    f1 = 0.99664  # N-S scale adjustment for WGS-84
    ncols_target = 1024


    location_info_keys = {
        "nrows", "ncols", "xllcorner", "yllcorner", "cellsize", "nodata_value"
    }

    def _generate_target_grid(self):
        """
        Generate longitudes and latitudes of the target grid

        """
        i1d = range(1024)
        j1d = range(1024)

        j2d, i2d = np.meshgrid(j1d, i1d)
        self.lons2d_target, self.lats2d_target = self.ij_to_lon_lat(i2d, j2d)


    def _generate_grid_from_descriptor(self, descr):
        """
        Assumes that the projection is the same as fir the 1024x1024 grid, but the size is different
        :param descr:
        """
        print(descr)
        nx, ny = descr
        i1d = range(nx)
        j1d = range(ny)

        j2d, i2d = np.meshgrid(j1d, i1d)
        return self.ij_to_lon_lat(i2d, j2d, ncols=nx)





    def __init__(self, data_folder="~/skynet3_rech1/nemo_obs_for_validation/ice_cover_glk/daily_grids/data_files"):
        """
        Note: be careful grids might be different in different files
        All indices start from 0
        :param data_folder:
        """
        self.data_folder = os.path.expanduser(data_folder)
        self.location_info_dict = {}

        # Domain descriptor is a tuple (nx, ny)
        self.domain_descr_to_kdtree = {}

        self.xllcorner = None
        self.yllcorner = None
        self.cellsize = None
        self.nrows = None
        self.ncols = None

        self.nodata_value = None

        # target coordinates
        self.lons2d_target = None
        self.lats2d_target = None

        self._generate_target_grid()

        # other lons and lats corresponding to the grid 516x510
        self.lons2d_other = None
        self.lats2d_other = None
        self.kdtree_other = None

        # Paths to other longitudes and latitudes
        self.path_to_other_lons = \
            "/RESCUE/skynet3_rech1/huziy/nemo_obs_for_validation/ice_cover_glk/daily_grids_1973_2002/Longrid.txt"
        self.path_to_other_lats = \
            "/RESCUE/skynet3_rech1/huziy/nemo_obs_for_validation/ice_cover_glk/daily_grids_1973_2002/Latgrid.txt"



    def lon_lat_to_ij(self, lon, lat, ncols=-1):
        """
        written based on the idl script from G. Lang
        :param lon:
        :param lat:
        """

        ncols = self.ncols_target if ncols < 0 else ncols

        lon = -lon
        alon0 = -self.wlon
        a1 = (ncols - 1.0) / np.radians(self.elon - self.wlon)
        y0 = a1 * np.log(np.tan(np.radians(45.0 + self.slat / 2.0)))

        ix = int(a1 * np.radians(alon0 - lon) + 0.5)
        jy = int(self.f1 * (a1 * np.log(np.tan(np.radians(lat / 2.0 + 45.0))) - y0) + 0.5)

        return ix, jy

    def ij_to_lon_lat(self, i, j, ncols=-1):
        """
        i and j are 0-based indices starting from the lower left corner of the grid
        supposed to work as inverse to lon_lat_to_ij
        :param i:
        :param j:
        """

        ncols = self.ncols_target if ncols < 0 else ncols

        alon0 = -self.wlon
        a1 = (ncols - 1.0) / np.radians(self.elon - self.wlon)
        y0 = a1 * np.log(np.tan(np.radians(45.0 + self.slat / 2.0)))

        lat = 2.0 * (np.degrees((np.arctan(np.exp((j / self.f1 + y0) / a1)))) - 45.0)
        lon = 1.0 * ((i / np.radians(a1)) - alon0)
        return lon, lat


    def generate_date_to_path_map(self):
        pass

    def read_location_info(self, fname):
        fpath = os.path.join(self.data_folder, fname)
        self.location_info_dict["file"] = fname

        with open(fpath) as f:
            self.location_info_dict["ncols"] = int(f.next().split()[1])
            self.location_info_dict["nrows"] = int(f.next().split()[1])
            self.location_info_dict["xllcorner"] = float(f.next().split()[1])
            self.location_info_dict["yllcorner"] = float(f.next().split()[1])
            self.location_info_dict["cellsize"] = float(f.next().split()[1])
            self.location_info_dict["nodata_value"] = int(f.next().split()[1])

            for key, val in self.location_info_dict.items():
                setattr(self, key, val)

    def get_location_info(self):
        return self.location_info_dict

    @staticmethod
    def date_to_fname(the_date):
        return "g{:%Y%m%d}.ct".format(the_date)


    @update_grid_info
    def get_icecover_interpolated_to(self, lons2d_target=None, lats2d_target=None,
                                     the_date=None,
                                     r_earth_m=6400e3):


        print(self.location_info_dict)

        lons2d_target[lons2d_target > 180] -= 360

        lats2d_target_r = np.radians(lats2d_target)
        lons2d_target_r = np.radians(lons2d_target)

        yt = r_earth_m * lats2d_target_r
        xt = r_earth_m * lons2d_target_r * np.cos(lats2d_target_r)

        print(xt.min(), xt.max())

        i0 = ((xt - self.xllcorner) / float(self.cellsize * np.cos(lats2d_target_r))).astype(int)
        j0 = ((yt - self.yllcorner) / float(self.cellsize)).astype(int)

        nxagg = (xt.max() - xt.min()) / (self.cellsize * xt.shape[0])
        nyagg = (yt.max() - yt.min()) / (self.cellsize * yt.shape[1])

        print("nxagg={}; nyagg={}".format(nxagg, nyagg))

        data_source = self.get_data_for_day(the_date=the_date)

        plt.figure()
        im = plt.pcolormesh(data_source.transpose())
        plt.colorbar(im)

        nx, ny = data_source.shape

        j0 = np.maximum(j0, 0)
        j0 = np.minimum(j0, ny - 1)

        print(i0.min(), i0.max())
        print(j0.min(), j0.max())

        i0 = np.maximum(i0, 0)
        i0 = np.minimum(i0, nx - 1)

        # imin = np.maximum(i0 - nxagg // 2, 0)
        # imax = np.minimum(i0 + nxagg // 2, nx - 1)
        # jmin = np.minimum(j0 - nyagg // 2, 0)
        # jmax = np.maximum(j0 + nyagg // 2, ny - 1)


        imin = imax = i0
        jmin = jmax = j0

        result = np.zeros_like(xt)
        result = result.flatten()
        imin = imin.flatten()
        imax = imax.flatten()
        jmin = jmin.flatten()
        jmax = jmax.flatten()

        for i in range(len(result)):
            result[i] = data_source[imin[i]:imax[i] + 1, jmin[i]:jmax[i] + 1]

        print(result.min(), result.max())
        result[np.isnan(result)] = np.ma.masked
        return result.reshape(xt.shape)


    @staticmethod
    def _parse_line(line):
        """
        Parse the line from the static file
        :param line:
        :return:
        """
        sarr = np.fromiter((c for c in line), dtype=np.dtype("S1"))
        sarr.shape = (len(line) // 3, 3)
        return [float(b"".join(s)) for s in sarr]

    def get_data_from_path(self, path):
        data = []
        nrows = None
        nodata_value = -1
        with open(path) as f:

            # Skip the first 6 lines
            last_position = -1
            line = ""
            for i in range(6):
                last_position = f.tell()
                line = f.readline()
                if "nrows" in line.lower():
                    nrows = int(line.split()[-1])

                if "nodata_value" in line.lower():
                    nodata_value = int(line.split()[-1])

            # unread the last line
            if line.split()[0].lower() not in self.location_info_keys:
                f.seek(last_position)


            for i, line in enumerate(f):
                line = line.rstrip()
                if line.strip() == "":
                    break


                data.insert(0, self._parse_line(line))

                # Exit if read all the rows
                # (Strange thing happens at the end of the file, maybe because it was created on windows.)
                if i == nrows - 1:
                    break


        data = np.asarray(data, dtype="f4")

        print("Data shape in file: {}".format(data.shape))

        return np.ma.masked_where((data == nodata_value) | (data == -1), data).transpose()


    def get_data_for_day(self, the_date):
        """
        :param the_date:
        :return: masked array at land and nodata points
        """
        path = os.path.join(self.data_folder, self.date_to_fname(the_date=the_date))
        return self.get_data_from_path(path)


    def get_data_from_file_interpolate_if_needed(self, the_path):

        the_path = str(the_path)

        data = self.get_data_from_path(the_path)
        if data.shape != (self.ncols_target, self.ncols_target):
            # The interpolation is needed
            domain_descr = data.shape
            if domain_descr not in self.domain_descr_to_kdtree:

                if domain_descr == (516, 510):
                    self.lons2d_other = np.flipud(np.loadtxt(self.path_to_other_lons)).transpose()
                    self.lats2d_other = np.flipud(np.loadtxt(self.path_to_other_lats)).transpose()
                else:
                    self.lons2d_other, self.lats2d_other = self._generate_grid_from_descriptor(domain_descr)

                xs, ys, zs = lat_lon.lon_lat_to_cartesian(self.lons2d_other.flatten(), self.lats2d_other.flatten())

                kdtree_other = cKDTree(data=list(zip(xs, ys, zs)))

                self.domain_descr_to_kdtree[data.shape] = kdtree_other

            kdtree_other = self.domain_descr_to_kdtree[data.shape]
            xt, yt, zt = lat_lon.lon_lat_to_cartesian(self.lons2d_target.flatten(), self.lats2d_target.flatten())
            dsts, inds = kdtree_other.query(list(zip(xt, yt, zt)))


            return data.flatten()[inds].reshape(self.lons2d_target.shape)

        else:
            return data


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    glerl_manager = GLERLIceCoverManager()