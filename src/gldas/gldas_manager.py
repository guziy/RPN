from datetime import datetime
from netCDF4 import Dataset
import os
from matplotlib.dates import date2num
from scipy.spatial.kdtree import KDTree
from data.timeseries import TimeSeries
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np


class GldasManager():
    def __init__(self, folder_path="/home/huziy/skynet3_exec1/gldas_data"):
        """
        Data access interface to the folder of netcdf files
        runoff units: kg/m^2/s = mm/s
        """
        self.data_folder = folder_path
        self.surface_rof_varname = "Qs_GDS0_SFC_ave4h"
        self.subsurface_rof_varname = "Qsb_GDS0_SFC_ave4h"

        self.date_format = "%m/%d/%Y (%H:%M)"
        self._init_date_to_path_dict()
        self._init_kd_tree()

        pass


    def plot_subsrof_ts(self, i=0, j=0):
        all_dates = list(sorted(self.date_to_path.keys()))
        vals = [self.get_field_for_date(x, var_name=self.subsurface_rof_varname)[i, j] for x in all_dates]
        vals1 = [self.get_field_for_date(x, var_name=self.surface_rof_varname)[i, j] for x in all_dates]

        print(min(vals), max(vals))
        dates_num = date2num(all_dates)
        print(min(dates_num), max(dates_num))
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(dates_num, vals, label="subsurf rof")
        plt.plot(dates_num, vals1, label="surf rof")
        plt.legend()
        #plt.xticks(rotation='vertical')
        plt.show()


    def _init_date_to_path_dict(self):
        self.date_to_path = {}
        for fName in os.listdir(self.data_folder):
            if not fName.endswith(".nc"): continue #regard only nectdf files
            path = os.path.join(self.data_folder, fName)
            ds = Dataset(path)
            srofVar = ds.variables[self.surface_rof_varname]
            date = datetime.strptime(srofVar.initial_time, self.date_format)
            self.date_to_path[date] = path
            ds.close()


    def _init_kd_tree(self):
        """
        Has to be called after self._init_date_to_path_dict
        """
        if not len(self.date_to_path):
            print("You should call {0} first".format("self._init_date_to_path_dict"))
            raise Exception()

        for d, path in self.date_to_path.items():
            ds = Dataset(path)

            lons1d = ds.variables["g0_lon_1"][:]
            lats1d = ds.variables["g0_lat_0"][:]

            self.lats2d, self.lons2d = np.meshgrid(lats1d, lons1d)

            x, y, z = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
            self.kdtree = KDTree(list(zip(x, y, z)))
            return

        pass


    def get_field_for_month_and_year(self, var_name="", month=None, year=None):
        d1 = datetime(year=year, month=month, day=1)
        path = self.date_to_path[d1]
        ds = Dataset(path)
        return ds.variables[var_name][:]

    def get_field_for_date(self, the_date, var_name=""):
        path = self.date_to_path[the_date]
        ds = Dataset(path)
        data = ds.variables[var_name][:].transpose()  # transpose because I allways use (lon, lat) order of coordinates
        ds.close()
        return data


    def get_srof_spat_integrals_over_points_in_time(self, lons2d_target, lats2d_target, mask,
                                                    areas2d, start_date=None, end_date=None):
        return self._get_spatial_integrals_over_points_in_time(lons2d_target, lats2d_target, mask, areas2d,
                                                               start_date=start_date, end_date=end_date,
                                                               var_name=self.surface_rof_varname)

    def get_subsrof_spat_integrals_over_points_in_time(self, lons2d_target, lats2d_target, mask,
                                                       areas2d, start_date=None, end_date=None):
        return self._get_spatial_integrals_over_points_in_time(lons2d_target, lats2d_target, mask, areas2d,
                                                               start_date=start_date, end_date=end_date,
                                                               var_name=self.subsurface_rof_varname)


    def _get_spatial_integrals_over_points_in_time(self, lons2d_target, lats2d_target, mask,
                                                   areas2d, start_date=None, end_date=None, var_name=""):
        """
        i)  Interpolate to the grid (lons2d_target, lats2d_target)
        ii) Apply the mask to the interpoated fields and sum with coefficients from areas2d

        Note: the interpolation is done using nearest neighbor approach

        returns a timeseries of {t -> sum(Ai[mask]*xi[mask])(t)}
        """


        #interpolation
        x1, y1, z1 = lat_lon.lon_lat_to_cartesian(lons2d_target.flatten(), lats2d_target.flatten())

        dists, indices = self.kdtree.query(list(zip(x1, y1, z1)))

        mask1d = mask.flatten().astype(int)
        areas1d = areas2d.flatten()

        result = {}
        for the_date in list(self.date_to_path.keys()):
            if start_date is not None:
                if start_date > the_date: continue

            if end_date is not None:
                if end_date < the_date: continue

            data = self.get_field_for_date(the_date, var_name=var_name)
            result[the_date] = np.sum(data.flatten()[indices][mask1d == 1] * areas1d[mask1d == 1])

        times = list(sorted(result.keys()))
        values = [result[x] for x in times]
        print("nvals, min, max", len(values), min(values), max(values))
        return TimeSeries(time=times, data=values)


def main():
    gm = GldasManager()
    gm.plot_subsrof_ts(i=112, j=112)
    #TODO: implement
    pass


if __name__ == "__main__":
    main()
    print("Hello world")
  