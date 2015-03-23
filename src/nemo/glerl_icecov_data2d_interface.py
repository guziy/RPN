from datetime import datetime
from mpl_toolkits.basemap import Basemap

__author__ = 'huziy'

import os
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


class GLERLIceCoverManager(object):

    def __init__(self, data_folder="~/skynet3_rech1/nemo_obs_for_validation/ice_cover_glk/daily_grids/data_files"):
        self.data_folder = os.path.expanduser(data_folder)

        self.lons = None
        self.lats = None

        self.ncols = None
        self.nrows = None
        self.xllcorner = None
        self.yllcorner = None
        self.cellsize = None
        self.nodata_value = None

        self._get_location_info()

        self.date_to_path = None

    def generate_date_to_path_map(self):
        pass


    def _get_location_info(self):
        for fn in os.listdir(self.data_folder):
            fpath = os.path.join(self.data_folder, fn)
            with open(fpath) as f:

                self.ncols = int(f.next().split()[1])
                self.nrows = int(f.next().split()[1])
                self.xllcorner = float(f.next().split()[1])
                self.yllcorner = float(f.next().split()[1])
                self.cellsize = float(f.next().split()[1])
                self.nodata_value = int(f.next().split()[1])

            return



    def get_clim_of_max_icecover_for_month(self, month, start_year, end_year):
        data_for_month = []
        for the_year in range(start_year, end_year + 1):
            prefix = "g{}{:02d}".format(the_year, month)
            data_for_month_and_for_year = []
            for fn in [f for f in os.listdir(self.data_folder) if f.startswith(prefix)]:
                fpath = os.path.join(self.data_folder, fn)
                data_for_month_and_for_year.append(self.get_data_from_path(fpath))

            data_for_month.append(np.ma.max(data_for_month_and_for_year, axis=0))
        return np.ma.max(data_for_month, axis=0).transpose()



    def get_clim_of_max_icecover_interpolated_to(self, lons2d_target=None, lats2d_target=None,
                                                 month=1, start_year=None, end_year=None,
                                                 r_earth_m=6400e3):
        lons2d_target[lons2d_target > 180] -= 360

        lons2d_target_r = np.radians(lons2d_target)
        lats2d_target_r = np.radians(lats2d_target)

        yt = r_earth_m * lats2d_target_r
        xt = r_earth_m * lons2d_target_r * np.cos(lats2d_target_r)

        print xt.min(), xt.max()
        print yt.min(), yt.max()

        i0 = ((xt - self.xllcorner) / self.cellsize).astype(int)
        j0 = ((yt - self.yllcorner) / self.cellsize).astype(int)

        nxagg = (xt.max() - xt.min()) / (self.cellsize * xt.shape[0])
        nyagg = (yt.max() - yt.min()) / (self.cellsize * yt.shape[1])

        print "nxagg={}; nyagg={}".format(nxagg, nyagg)

        data_source = self.get_clim_of_max_icecover_for_month(month=month, start_year=start_year, end_year=end_year)

        plt.figure()
        plt.pcolormesh(data_source.transpose())
        plt.show()

        nx, ny = data_source.shape


        j0 = np.maximum(j0, 0)
        j0 = np.minimum(j0, ny - 1)

        i0 = np.maximum(i0, 0)
        i0 = np.minimum(i0, nx - 1)


        imin = np.maximum(i0 - nxagg // 2, 0)
        imax = np.minimum(i0 + nxagg // 2, nx - 1)
        jmin = np.minimum(j0 - nyagg // 2, 0)
        jmax = np.maximum(j0 + nyagg // 2, ny - 1)

        result = np.ma.masked_all_like(xt)
        result = result.flatten()
        imin = imin.flatten()
        imax = imax.flatten()
        jmin = jmin.flatten()
        jmax = jmax.flatten()

        for i in range(len(result)):
            result[i] = data_source[imin[i]:imax[i] + 1, jmin[i]:jmax[i] + 1].mean()

        print result.min(), result.max(())

        plt.figure()
        plt.pcolormesh(result.reshape(xt.shape).transpose())
        plt.show()






    def get_data_from_path(self, path):
        data = []
        with open(path) as f:
            for i in range(6):
                f.next()

            for i, line in enumerate(f):


                line = line.rstrip()
                if line == "":
                    break

                sarr = np.fromiter((c for c in line), dtype=np.dtype("S1"))

                sarr.shape = (len(line) // 3, 3)
                data.insert(0, [int("".join(s)) for s in sarr])

        data = np.asarray(data)

        assert data.shape == (self.nrows, self.ncols)
        return np.ma.masked_where((data == self.nodata_value) | (data == -1), data)



    def get_data_for_day(self, the_date):
        """
        :param the_date:
        :return: masked array at land and nodata points
        """
        path = os.path.join(self.data_folder, "g{:%Y%m%d}.ct".format(the_date))
        return self.get_data_from_path(path)


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()





    glerl_manager = GLERLIceCoverManager()
    # glerl_manager.get_data_for_day(the_date=datetime(2005, 1, 3))
    glerl_manager.get_clim_of_max_icecover_interpolated_to(

    )