
import numpy as np
from rpn.domains import lat_lon
from scipy.spatial import KDTree


class SubRegionByLonLatCorners(object):

    def __init__(self, lleft: dict, uright: dict):
        """
        Region defined by 2 corner poits in lat lon, but the sides are orthogonal in the grid projection
        """

        self.lleft = lleft
        self.lleft_lon = lleft["lon"]
        self.lleft_lat = lleft["lat"]

        self.uright = uright
        self.uright_lon = uright["lon"]
        self.uright_lat = uright["lat"]


    def to_mask(self, lons_2d_grid, lats_2d_grid):

        """

        :param lons_2d_grid:
        :param lats_2d_grid:
        :return: the mask of the subregion corresponding to the grid with the upper right and lower left points from self
        """
        x_g, y_g, z_g = lat_lon.lon_lat_to_cartesian(lons_2d_grid.flatten(), lats_2d_grid.flatten())
        ktree = KDTree(list(zip(x_g, y_g, z_g)))

        ll_x, ll_y, ll_z = lat_lon.lon_lat_to_cartesian(self.lleft_lon, self.lleft_lat)
        ur_x, ur_y, ur_z = lat_lon.lon_lat_to_cartesian(self.uright_lon, self.uright_lat)


        i_g, j_g = np.indices(lons_2d_grid.shape)

        i_g_flat, j_g_flat = i_g.flatten(), j_g.flatten()


        _, ind_ll = ktree.query((ll_x, ll_y, ll_z), k=1)
        _, ind_ur = ktree.query((ur_x, ur_y, ur_z), k=1)


        i_ll, j_ll = i_g_flat[ind_ll], j_g_flat[ind_ll]
        i_ur, j_ur = i_g_flat[ind_ur], j_g_flat[ind_ur]

        res = np.zeros_like(lons_2d_grid, dtype=bool)
        res[i_ll:i_ur + 1, j_ll: j_ur + 1] = 1

        return res, (i_ll, j_ll), (i_ur, j_ur)
