from datetime import timedelta, datetime
import itertools
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.spatial.kdtree import KDTree
import application_properties

from permafrost import draw_regions
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt


class CRUDataManager:
    def __init__(self, path = "data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.tmp.dat.nc", var_name = "tmp"):
        ds = Dataset(path)
        self.var_name = var_name
        self._init_fields(ds)
        self.nc_dataset = ds

        pass

    def _init_fields(self, nc_dataset):
        nc_vars = nc_dataset.variables
        lons = nc_vars["lon"][:]
        lats = nc_vars["lat"][:]

        lats2d, lons2d = np.meshgrid(lats, lons)

        self.lons2d, self.lats2d = lons2d, lats2d

        times = nc_vars["time"][:]
        time_units_s = nc_vars["time"].units

        step_s, start_date_s = map(lambda x: x.strip(), time_units_s.split("since"))

        start_date = datetime.strptime(start_date_s, "%Y-%m-%d")
        if step_s == "hours":
            self.times = map(lambda x: start_date + timedelta(minutes = x * 60), times )
        elif step_s == "days":
            self.times = map(lambda x: start_date + timedelta(minutes = x * 60 * 24), times )


        self.var_data = np.transpose( nc_vars[self.var_name][:], axes=(0,2,1))



    def get_mean(self, start_year, end_year, months = None):
        """
        returns the mean for the period [start_year, end_year], over the months
        :type months: list
        months = list of month numbers over which the averaging is done
        """

        sel_times = itertools.ifilter(lambda x: (start_year <= x.year) and (x.year <= end_year), self.times)
        bool_vector = np.array(map( lambda x: x.month in months, sel_times))
        return np.mean(self.var_data[bool_vector, :, :], axis=0)



    def interpolate_data_to(self, data_in, lons2d, lats2d, nneighbours = 4):
        """
        Interpolates data_in to the grid defined by (lons2d, lats2d)
        assuming that the data_in field is on the initial CRU grid

        interpolate using 4 nearest neighbors and inverse of squared distance
        """
        x_in,y_in,z_in = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
        kdtree = KDTree(zip(x_in, y_in, z_in))

        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())

        dst, ind = kdtree.query(zip(x_out, y_out, z_out), k=nneighbours)

        data_in_flat = data_in.flatten()

        inverse_square = 1.0 / dst ** 2
        if len(dst.shape) > 1:
            norm = np.sum(inverse_square, axis=1)
            norm = np.array( [norm] * dst.shape[1] ).transpose()
            coefs = inverse_square / norm

            data_out_flat = np.sum( coefs * data_in_flat[ind], axis= 1)
        elif len(dst.shape) == 1:
            data_out_flat = data_in_flat[ind]
        else:
            raise Exception("Could not find neighbor points")
        return np.reshape(data_out_flat, lons2d.shape)


def main():
    dm = CRUDataManager()

    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()

    x, y = b(dm.lons2d, dm.lats2d)


    fig = plt.figure()

    gs = gridspec.GridSpec(1,2)
    ax = fig.add_subplot(gs[0,0])
    data = dm.get_mean(1981, 2009, months = [6,7,8])
    img = b.contourf(x, y, data.copy(), ax = ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax)
    b.drawcoastlines(ax = ax)
    ax.set_title("CRU (not interp.), \n JJA period: {0} - {1}".format(1981, 2009))


    ax = fig.add_subplot(gs[0,1])
    data_projected = dm.interpolate_data_to(data, lons2d, lats2d)
    x, y = b(lons2d, lats2d)
    img = b.contourf(x, y, data_projected, ax = ax, levels = img.levels)

    #add pretty colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax)

    b.drawcoastlines(ax = ax)
    ax.set_title("CRU ( interp.), \n JJA period: {0} - {1}".format(1981, 2009))

    plt.savefig("t_cru_jja.png")


    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  