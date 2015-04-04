from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from mpl_toolkits.basemap import Basemap
from pandas.core.series import TimeSeries
from scipy.spatial.kdtree import KDTree
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

#Plots a timeseries of a point picked on a map

class TimeSeriesPlotter:
    def __init__(self, ax, basemap, lons2d, lats2d, ncVarDict, times, start_date, end_date):
        """
        Plots a vertical profile at the point nearest to the clicked one
        :type ax: Axes
        """
        assert isinstance(ax, Axes)
        self.basemap = basemap
        assert isinstance(self.basemap, Basemap)

        self.lons_flat = lons2d.flatten()
        self.lats_flat = lats2d.flatten()
        self.ncVarDict = ncVarDict
        self.lons2d = lons2d
        self.lats2d = lats2d
        self.counter = 0
        self.ax = ax
        x, y, z = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
        self.kdtree = KDTree(list(zip(x,y,z)))

        self.sel_time_indices = np.where([start_date <= t <= end_date for t in times])[0]
        self.times = times[self.sel_time_indices]


        ax.figure.canvas.mpl_connect("button_press_event", self)


    def _get_closest_ij(self, event):
        lon, lat = self.basemap(event.xdata, event.ydata, inverse = True)

        x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        dist, i = self.kdtree.query((x0,y0,z0))

        lon0, lat0 = self.lons_flat[i], self.lats_flat[i]

        ind = np.where((self.lons2d == lon0) & (self.lats2d == lat0))


        ix = ind[0][0]
        jy = ind[1][0]
        return ix, jy



    def _plot_timeseries(self, ax, ix, jy):
        fig_daily = plt.figure()
        ax_daily = plt.gca()

        fig_monthly = plt.figure()
        ax_monthly = plt.gca()


        for varName, ncVar in self.ncVarDict.items():
            sel_values = ncVar[self.sel_time_indices, 0, ix, jy]
            ax.plot(self.times, sel_values, label = varName)

            #calculate and plot daily means
            ts = pd.TimeSeries(index = self.times, data = sel_values)
            ts = ts.resample("D", how = "mean")
            ax_daily.plot(ts.index, ts.values, label = varName)

            #calculate and plot monthly means
            ts = ts.resample("M", how = "mean")
            ax_monthly.plot(ts.index, ts.values, label = varName)



        ax.legend()
        ax.set_title(str(self.counter))

        ax_daily.legend()
        ax_daily.set_title(str(self.counter) + " - daily")


        ax_monthly.legend()
        ax_monthly.set_title(str(self.counter) + " - monthly")
        assert isinstance(ax, Axes)





    def __call__(self, event):
        print(event.xdata, event.ydata)

        print(event.button)
        if event.button != 3:
            return
        ix, jy = self._get_closest_ij(event)

        fig = plt.figure()
        sounding_ax = fig.add_subplot(1,1,1)
        self._plot_timeseries(sounding_ax, ix, jy)

        self.ax.annotate(str(self.counter), (event.xdata, event.ydata), font_properties =
                FontProperties(size=10))
        self.ax.redraw_in_frame()

        self.counter += 1
        assert isinstance(fig, Figure)
        plt.show()



def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  