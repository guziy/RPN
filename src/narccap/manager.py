from datetime import datetime
import os
import pickle
from scipy.spatial.kdtree import KDTree
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from netCDF4 import MFDataset,  num2date
from netCDF4 import date2num

# mrro = srof + ssrof (total runoff)
#mrros = srof (surface runoff)

class NarccapDataManager():

    def __init__(self, var_names = None, folder_with_nc_data = "/home/huziy/skynet1_rech3/narccap"):
        """
        Each time when reading a given simulation the coordinates values should be taken frm the data
        for the current simulation, since it changes from simulation to simulation
        """
        self.var_names = ["mrro", "mrros"] if var_names is None else var_names #variables of interest
        self.monthlyMeanFields = None
        self.annualMeanFields = None
        self.folder_with_nc_data = folder_with_nc_data
        self.cache_files_folder = "/home/huziy/skynet1_rech3/narccap/climatologies"

        self.lon_name = "lon"
        self.lat_name = "lat"







    def _get_clim_cache_file_path(self, varname = "mrro", gcm = "", rcm = "",
                                             start_year = None, end_year = None,
                                             months = None):
        """
        When months is None, then annual climatology is calculated
        """
        if months is None:
            months = range(1,13)

        return "{0}_{1}-{2}_{3}_{4}_{5}.bin".format(varname, gcm, rcm, start_year, end_year, "_".join(map(str, months)))

    def get_climatologic_field(self,  varname = "mrro", gcm = "", rcm = "",
                                         start_year = None, end_year = None,
                                         months = None
                                         ):
        """
             for time t: start_year <= t <= end_year
        """

        mfds = MFDataset("{0}/{1}-{2}/current/{3}_*.nc".format(self.folder_with_nc_data, gcm, rcm, varname))

        self.lon2d = mfds.variables[self.lon_name][:].transpose()
        self.lat2d = mfds.variables[self.lat_name][:].transpose()
        self._init_kd_tree()



        cache_file = self._get_clim_cache_file_path(varname = varname, gcm=gcm, rcm = rcm,
            start_year=start_year, end_year=end_year, months=months)


        cache_file = os.path.join(self.cache_files_folder, cache_file)

        if os.path.isfile(cache_file):
            f = open(cache_file)
            mfds.close()
            return pickle.load(f)




        t = mfds.variables["time"]
        t_units = t.units
        t_calendar = t.calendar

        t_start = date2num(datetime(start_year, 1,1), t_units, calendar=t_calendar)
        t_end = date2num(datetime(end_year+1, 1,1), t_units, calendar=t_calendar)

        t = t[:]
        t_sel = t[(t_start <= t) & (t < t_end)]
        dates_sel = num2date(t_sel, t_units, calendar=t_calendar)

        bool_vect = np.array( map(lambda x: x.month in months, dates_sel), dtype=np.bool )
        data_sel = mfds.variables[varname][ (t_start <= t) & (t < t_end),:,:]


        #save results to a cache file for reuse
        result = data_sel[bool_vect,:,:].mean(axis = 0).transpose()
        pickle.dump(result, open(cache_file,"w"))
        mfds.close()
        return result #because in the file the axes are inversed





    def inerpolate_to(self, model_lons, model_lats, data, nneighbors = 1):
        """
        Interpolate data to the grid of model_lons and model_lats (2D fields)
        """
        x, y, z = lat_lon.lon_lat_to_cartesian(model_lons.flatten(), model_lats.flatten())

        d, inds = self.kdtree.query(zip(x, y, z), k = nneighbors)

        if nneighbors == 1:
           data1d = data.flatten()
           print "interpolation inds = ", inds
           print model_lats.shape, len(inds)
           print data.shape
           return data1d[inds].reshape(model_lons.shape)
        else:
            raise NotImplementedError("The interpolation is not yet implemented for more than one neighbor.")


        pass


    def _init_kd_tree(self):
        """
        Init KDTree used for interpolation
        """
        x0, y0, z0 = lat_lon.lon_lat_to_cartesian(self.lon2d.flatten(), self.lat2d.flatten())
        self.kdtree = KDTree(zip(x0, y0, z0))


def main():
    import  matplotlib.pyplot as plt
    nObj = NarccapDataManager()
    x = nObj.get_climatologic_field(start_year=1985, end_year=1990, months=[1,2], gcm="ccsm", rcm="crcm")
    print x.shape, x.min(), x.max()
    plt.figure()
    plt.pcolormesh(x.transpose())
    plt.show()

    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  