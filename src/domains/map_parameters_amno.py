from scipy.spatial.kdtree import KDTree
from util.geo import lat_lon

__author__="huziy"
__date__ ="$15 nov. 2010 23:28:35$"

import numpy as np
from mpl_toolkits.basemap import Basemap
from util.geo.ps_and_latlon import psxy2latlon, latlon2psxy


class MapParameters():
    def __init__(self):
        self.n_cols = 180
        self.n_rows = 172
        self.lon_center = None
        self.lat_center = None

        self.x_min = None
        self.y_min = None

        self.xc = None
        self.yc = None

        self.dx = None
        self.dy = None

        self._kdtree = None

        #initialize fields
        [self.xs, self.ys, self.basemap] = self.init_map()
        assert isinstance(self.basemap, Basemap)
   



    def get_kd_tree(self):
        """
        :rtype : KDTree
        for interpolation purposes
        """
        if self._kdtree is None:
            x, y, z = lat_lon.lon_lat_to_cartesian(self.lons.flatten(), self.lats.flatten() )
            self._kdtree = KDTree(list(zip(x,y,z)))
        return self._kdtree

    def get_indices_of_the_closest_point_to(self, lon, lat):
        """
        get indices of the point closest to the coord (lon, lat)
        """
        [x, y] = latlon2psxy(lat, lon)
        return round(x - self.x_min), round(y - self.y_min)


    def get_longitudes_and_latitudes(self, nx = 180, ny = 172,
                                     lat_center = 49.65698,
                                     lon_center = -96.99443, dx = 45000):
        """
        get longitudes and latitudes of the AMNO grid
        """

        #coordinates with respect to the pole, divided by
        #dx
        [xc, yc] = latlon2psxy(lat_center, lon_center)

        self.lon_center = lon_center
        self.lat_center = lat_center

        self.dx = dx
        self.dy = dx

        print('Coordinates of the grid center ', xc * dx, yc * dx)
        

        xmin = xc - (nx - 1) / 2.0
        ymin = yc - (ny - 1) / 2.0

        self.x_min = xmin
        self.y_min = ymin

        print('These coordinates can be verified with cccma site points (2,2) and (181, 173) respectively')
        print('lower left: ', psxy2latlon(xmin, ymin))
        print('upper right: ', psxy2latlon(xmin + nx - 1 , ymin + ny - 1))

        longitudes = np.zeros((nx, ny))
        latitudes = np.zeros((nx, ny))

        for i in range(nx):
            for j in range(ny):
                latitudes[i,j], longitudes[i, j] = psxy2latlon(xmin + i, ymin + j)

        return longitudes, latitudes


    def init_map(self):
        """initializes longitudes and latitudes of grid"""
        [self.lons, self.lats] = self.get_longitudes_and_latitudes( self.n_cols, self.n_rows)
        m = Basemap(projection = 'npstere', #area_thresh = 10000,
                        lat_ts = 60, lat_0 = 60, lon_0 = -115, boundinglat = 0, resolution='i')

#        m = Basemap(projection = 'npstere',
#                        lat_ts = 60, lat_0 = -10, lon_0 = -90, boundinglat = 40, resolution='i')
#

        [xs, ys] = m(self.lons, self.lats)
        return xs, ys, m

    def get_resolution_meters(self):
        return self.dx


def zoom_on_quebec(plt):
    [ymin, ymax] = plt.ylim()
    plt.ylim(ymin + 0.12 * (ymax - ymin), ymax * 0.32)

    [xmin, xmax] = plt.xlim()
    plt.xlim(xmin + (xmax - xmin) * 0.65, 0.85*xmax)


polar_stereographic = MapParameters()

def test():
    polar_stereographic = MapParameters()
    clon = polar_stereographic.lon_center
    clat = polar_stereographic.lat_center
    print(clon, clat)
    [i, j] = polar_stereographic.get_indices_of_the_closest_point_to(clon, clat)
    print(i,j)

if __name__ == "__main__":
    test()
    print("Hello World")
