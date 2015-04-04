import os
from mpl_toolkits.basemap import Basemap
from scipy.spatial.kdtree import KDTree
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt

class GrdcDataManager:
    def __init__(self, path_tofolder = "/home/huziy/skynet3_exec1/grdc_global"):

        self.path_to_annual_rof = os.path.join(path_tofolder, "obs_ro.grd")
        self.lons2d = None
        self.lats2d = None

        self.ncols = None
        self.nrows = None
        self.xll = None
        self.yll = None

        self.cellsize = None
        self.nodata_value = None



        self.ktree = None
        pass


    def _get_lon_lats(self):
        if self.lons2d is None:
            lons = [ self.xll + i * self.cellsize for i in range(self.ncols) ]
            lats = [ self.yll + i * self.cellsize for i in range(self.nrows) ]
            self.lats2d, self.lons2d = np.meshgrid(lats, lons)


        return self.lons2d, self.lats2d



    def interpolate_data_to_model_grid(self, model_lons_2d, model_lats_2d, data_obs):
        x0, y0, z0 = lat_lon.lon_lat_to_cartesian(model_lons_2d.flatten(), model_lats_2d.flatten())
        x, y, z = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())

        if self.ktree is None:
            self.ktree = KDTree(list(zip(x, y, z)))

        d, i = self.ktree.query(list(zip(x0, y0, z0)))

        return data_obs.flatten()[i].reshape(model_lons_2d.shape)




    def _read_data_from_file(self, path):
        f = open(path)
        vals = []
        for line in f:
            line = line.strip()
            if line == "": continue

            if line.startswith("ncols"):
                self.ncols = int(line.split()[1].strip())
            elif line.startswith("nrows"):
                self.nrows = int(line.split()[1].strip())
            elif line.startswith("xllcorner"):
                self.xll = float(line.split()[1].strip())
            elif line.startswith("yllcorner"):
                self.yll = float(line.split()[1].strip())
            elif line.startswith("cellsize"):
                self.cellsize = float(line.split()[1].strip())
            elif line.startswith("NODATA"):
                self.nodata_value = int(line.split()[1].strip())
            else:
                vals.append( list(map(float, [s.strip() for s in line.split()])) )
                #print len(vals), self.ncols * self.nrows




        vals = np.array( vals[::-1] )#reverse row order
        vals = vals.transpose()
        return vals




    def get_mean_annual_runoff_in_mm_per_s(self):
        vals = self._read_data_from_file(self.path_to_annual_rof)
        #vals = np.ma.masked_where(vals.astype(int) == self.nodata_value, vals)
        vals = np.ma.masked_where(vals < 0, vals)

        print(self.nodata_value, np.min(vals), self.nodata_value == np.min(vals))

        vals /= 365 * 24 * 60 * 60 #convert to mm/s

        self._get_lon_lats()
        return self.lons2d, self.lats2d, vals


def main():
    #TODO: implement

    fig = plt.figure()
    obs_manager = GrdcDataManager()
    b = Basemap()
    lons, lats, data = obs_manager.get_mean_annual_runoff_in_mm_per_s()
    x, y = b(lons, lats)
    #qm = b.pcolormesh(x, y, data)
    img = b.contourf(x, y , data)
    b.drawcoastlines()
    fig.colorbar(img)

    plt.show()


    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  