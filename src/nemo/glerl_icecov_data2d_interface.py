from datetime import datetime
from mpl_toolkits.basemap import Basemap

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
    def __init__(self, data_folder="~/skynet3_rech1/nemo_obs_for_validation/ice_cover_glk/daily_grids/data_files"):
        """
        Note: be careful grids might be different in different files
        :param data_folder:
        """
        self.data_folder = os.path.expanduser(data_folder)
        self.location_info_dict = {}

        self.xllcorner = None
        self.yllcorner = None
        self.cellsize = None
        self.nrows = None
        self.ncols = None

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

            for key, val in self.location_info_dict.iteritems():
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



        print self.location_info_dict


        lons2d_target[lons2d_target > 180] -= 360


        lats2d_target_r = np.radians(lats2d_target)
        lons2d_target_r = np.radians(lons2d_target)


        yt = r_earth_m * lats2d_target_r
        xt = r_earth_m * lons2d_target_r * np.cos(lats2d_target_r)

        print xt.min(), xt.max()


        i0 = ((xt - self.xllcorner) / float(self.cellsize)).astype(int)
        j0 = ((yt - self.yllcorner) / float(self.cellsize)).astype(int)

        nxagg = (xt.max() - xt.min()) / (self.cellsize * xt.shape[0])
        nyagg = (yt.max() - yt.min()) / (self.cellsize * yt.shape[1])

        print "nxagg={}; nyagg={}".format(nxagg, nyagg)

        data_source = self.get_data_for_day(the_date=the_date)

        plt.figure()
        im = plt.pcolormesh(data_source.transpose())
        plt.colorbar(im)

        nx, ny = data_source.shape


        j0 = np.maximum(j0, 0)
        j0 = np.minimum(j0, ny - 1)

        print i0.min(), i0.max()
        print j0.min(), j0.max()

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

        print result.min(), result.max()
        result[np.isnan(result)] = np.ma.masked
        return result.reshape(xt.shape)


    def get_data_from_path(self, path):
        data = []
        print path
        with open(path) as f:

            # Skip the first 6 lines
            for i in range(6):
                f.next()



            for i, line in enumerate(f):


                line = line.rstrip()
                if line == "":
                    break

                sarr = np.fromiter((c for c in line), dtype=np.dtype("S1"))


                sarr.shape = (len(line) // 3, 3)
                data.insert(0, [int("".join(s)) for s in sarr])

                # Exit if read all the rows
                # (Strange thing happens at the end of the file, maybe because it was created on windows.)
                if i == self.nrows - 1:
                    break

        data = np.asarray(data)

        assert data.shape == (self.nrows, self.ncols)
        return np.ma.masked_where((data == self.nodata_value) | (data == -1), data).transpose()


    def get_data_for_day(self, the_date):
        """
        :param the_date:
        :return: masked array at land and nodata points
        """
        path = os.path.join(self.data_folder, self.date_to_fname(the_date=the_date))
        return self.get_data_from_path(path)


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    glerl_manager = GLERLIceCoverManager()