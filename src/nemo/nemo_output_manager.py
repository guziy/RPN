from netCDF4 import num2date
import os
import datetime
from iris.cube import Cube
from matplotlib import cm
from matplotlib.colors import LogNorm, BoundaryNorm
from mpl_toolkits.basemap import maskoceans
import numpy as np
from scipy.spatial.ckdtree import cKDTree
from nemo import nemo_commons
from util.geo import lat_lon

__author__ = 'huziy'

import iris
import pandas as pd

from iris.analysis import cartography


class NemoOutputManager(object):

    def __init__(self, file_path = "", var_name = "",
                 bathymetry_path = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_Michigan/EXP00/bathy_meter.nc"):
        """
        :param file_path:
        :param var_name:
        :param bathymetry_path: used to mask land points
        """
        self.current_time_frame = -1
        self.var_name = var_name

        self.cube = iris.load_cube(file_path, constraint=iris.Constraint(cube_func=lambda c: c.var_name == var_name))
        self.lons, self.lats = cartography.get_xy_grids(self.cube)

        lons2d_gl, lats2d_gl = nemo_commons.get_2d_lons_lats_from_nemo(path=bathymetry_path)
        mask_gl = nemo_commons.get_mask(path=bathymetry_path)

        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons2d_gl.flatten(), lats2d_gl.flatten())
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(self.lons.flatten(), self.lats.flatten())

        tree = cKDTree(zip(xs, ys, zs))
        dists, indices = tree.query(zip(xt, yt, zt))

        self.mask = mask_gl.flatten()[indices].reshape(self.lons.shape)


        self.nt = self.cube.shape[0]
        assert isinstance(self.cube, Cube)
        print self.nt


    def get_total_number_of_time_frames(self):
        return self.nt


    def get_area_mean_timeseries(self):
        """
        :return: pandas timeseries
        """

        if self.cube.ndim == 3:
            surface_data = np.ma.masked_array(self.cube.data)
        else:
            raise Exception("Do not know how to handle {}D data".format(self.cube.ndim))

        i_to_mask, j_to_mask = np.where(~self.mask)
        surface_data[:, i_to_mask, j_to_mask] = np.ma.masked

        ts_vals = surface_data.mean(axis=1).mean(axis=1)
        time = self.cube.coord("time")

        # print dir(time.units)
        dates = num2date(time.points[:], str(time.units))



        return pd.DataFrame(data=ts_vals, index=dates, columns=["NEMO", ])


    def get_next_time_frame_data(self, level = 0):
        self.current_time_frame += 1
        assert isinstance(self.cube, Cube)
        if self.cube.data.ndim == 4:
            selection = self.cube.data[self.current_time_frame, level, ...]
        elif self.cube.data.ndim == 3:
            selection = self.cube.data[self.current_time_frame, ...]
        else:
            raise ValueError("Do not know how to handle {} dimensional frames".format(self.cube.data.ndim))

        return selection


    def get_current_time(self):
        units = self.cube.coord("time").units
        return units.num2date(self.cube.coord("time").points[self.current_time_frame])


    def is_last_time_frame(self):
        pass


    def reinitialize_timeframe_counter(self):
        self.current_time_frame = 0


def check():
    import matplotlib.pyplot as plt

    data_folder = "/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_Michigan/GLK_LIM3_Michigan_1981-2000_dt30min"
    manager_u = NemoOutputManager(
        file_path = os.path.join(data_folder, "GLK_1d_19810101_20001231_grid_U.nc"),
        var_name="vozocrtx")

    manager_v = NemoOutputManager(
        file_path=os.path.join(data_folder, "GLK_1d_19810101_20001231_grid_V.nc"),
        var_name="vomecrty")

    manager_t = NemoOutputManager(
        file_path=os.path.join(data_folder, "GLK_1d_19810101_20001231_grid_T.nc"),
        var_name="sosstsst"
    )

    fig = plt.figure()
    from matplotlib import animation

    #data = (manager_u.get_next_time_frame() ** 2 + manager_v.get_next_time_frame() ** 2) ** 0.5
    data = manager_t.get_next_time_frame_data()

    #data = np.ma.masked_where(data <= 0, data)

    #bounds = [0, 0.02, 0.05, 0.08, 0.09, 0.1]
    bounds = np.arange(-20, 21, 1)
    bn = BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)
    #img = plt.pcolormesh(data, vmin = bounds[0], vmax = bounds[-1], cmap = cm.get_cmap("jet", len(bounds) - 1), norm = bn)
    qp = plt.quiver(manager_u.get_next_time_frame_data(), manager_v.get_next_time_frame_data(), scale=1.5)
    #plt.colorbar()

    def update_fig(*args):
        print manager_u.get_current_time()
        u = manager_u.get_next_time_frame_data()
        v = manager_v.get_next_time_frame_data()
        cur_data = (u ** 2 + v ** 2) ** 0.5
        cur_data = manager_t.get_next_time_frame_data()
        #cur_data = np.ma.masked_where(cur_data <= 0, cur_data)
        print cur_data.min(), cur_data.max()
        qp.set_UVC(u.flatten(), v.flatten())
        plt.title(str(manager_u.current_time_frame))

        return qp,


    ani = animation.FuncAnimation(fig, update_fig, blit = False, frames = 100)
    plt.show()



if __name__ == "__main__":
    check()