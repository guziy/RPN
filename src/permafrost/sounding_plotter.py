from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from mpl_toolkits.basemap import Basemap
from scipy.spatial.kdtree import KDTree

__author__ = 'huziy'

import numpy as np
from util.geo import lat_lon

import matplotlib.pyplot as plt


class SoundingPlotter:
    def __init__(self, ax , basemap, tmin_3d, tmax_3d, lons2d, lats2d, levelheights = None):
        """
        Plots a vertical profile at the point nearest to the clicked one
        :type ax: Axes
        """
        assert isinstance(ax, Axes)
        self.basemap = basemap
        assert isinstance(self.basemap, Basemap)
        self.tmin_3d = tmin_3d
        self.tmax_3d = tmax_3d
        self.lons2d = lons2d
        self.lats2d = lats2d
        self.T0 = 273.15

        self.lons_flat = lons2d.flatten()
        self.lats_flat = lats2d.flatten()
        self.level_heights = levelheights

        self.counter = 0
        self.ax = ax
        x, y, z = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
        self.kdtree = KDTree(list(zip(x,y,z)))
        ax.figure.canvas.mpl_connect("button_press_event", self)
        pass


    def _get_closest_ij(self, event):
        lon, lat = self.basemap(event.xdata, event.ydata, inverse = True)

        x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        dist, i = self.kdtree.query((x0,y0,z0))

        lon0, lat0 = self.lons_flat[i], self.lats_flat[i]

        ind = np.where((self.lons2d == lon0) & (self.lats2d == lat0))


        ix = ind[0][0]
        jy = ind[1][0]
        return ix, jy



    def _plot_sounding(self, ax, ix, jy):
        ax.plot(self.tmax_3d[ix, jy, :] - self.T0, self.level_heights, color = "r")
        ax.plot(self.tmin_3d[ix, jy, :] - self.T0, self.level_heights, color = "b")
        ax.plot([0 , 0], [self.level_heights[0], self.level_heights[-1]], color = "k")
        ax.set_title(str(self.counter))

        assert isinstance(ax, Axes)
        ax.invert_yaxis()




    def __call__(self, event):
        print(event.xdata, event.ydata)

        print(event.button)
        if event.button != 3:
            return
        ix, jy = self._get_closest_ij(event)

        fig = plt.figure()
        sounding_ax = fig.add_subplot(1,1,1)
        self._plot_sounding(sounding_ax, ix, jy)

        self.ax.annotate(str(self.counter), (event.xdata, event.ydata), font_properties =
                FontProperties(size=10))
        self.ax.redraw_in_frame()

        self.counter += 1
        assert isinstance(fig, Figure)
        plt.show()



        pass



def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  