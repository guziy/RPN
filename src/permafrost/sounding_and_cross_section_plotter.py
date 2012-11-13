from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

__author__ = 'huziy'

import numpy as np
from sounding_plotter import SoundingPlotter
import matplotlib.pyplot as plt

class SoundingAndCrossSectionPlotter(SoundingPlotter):

    def __init__(self, ax, basemap, tmin_3d, tmax_3d, lons2d, lats2d, levelheights=None, temporal_data = None):
        """
        temporal_data is supposed to be a 4D numpy array with the dimensions in the following order:
        (t, x, y, z) - and this data is used to plot temporal_data = f(t, z) [update, actually I made the temporal_data,
         to be annual means, since it is not possible to hold that much of data in memory]
        """
        SoundingPlotter.__init__(self, ax, basemap, tmin_3d, tmax_3d, lons2d, lats2d, levelheights = levelheights)


        self.annual_means = temporal_data
        print self.annual_means.shape



    def _plot_cross_section(self, fig, ax, ix, jy):

        tvals = range(self.annual_means.shape[0])
        zvals = self.level_heights

        z, t = np.meshgrid(zvals, tvals)

        cs = ax.contourf(t, z, self.annual_means[:,ix, jy, :])
        ax.invert_yaxis()
        ax.set_title(str(self.counter))
        fig.colorbar(cs, ax = ax)


    def __call__(self, event):
        print event.xdata, event.ydata

        print event.button
        if event.button != 3:
            return
        ix, jy = self._get_closest_ij(event)

        fig = plt.figure()
        assert isinstance(fig, Figure)
        gs = gridspec.GridSpec(1,2)
        #sounding
        sounding_ax = fig.add_subplot(gs[0,0])
        self._plot_sounding(sounding_ax, ix, jy)

        #cross-section of temperature
        cross_ax = fig.add_subplot(gs[0,1])
        self._plot_cross_section(fig, cross_ax, ix, jy)

        self.ax.annotate(str(self.counter), (event.xdata, event.ydata), font_properties =
                FontProperties(size=10),  bbox=dict(boxstyle="round", fc="w"))

        assert isinstance(fig, Figure)
        fig.savefig("pf_images/image_cross_and_profile_{0}.png".format(self.counter))
        self.counter += 1

        self.ax.figure.show()
        fig.show()
        #self.ax.figure.canvas.draw()


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  
