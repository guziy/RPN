from matplotlib.colors import Normalize
import numpy as np
__author__ = 'huziy'


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        """
        Copied from SO answer: http://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
        :param vmin:
        :param vmax:
        :param midpoint:
        :param clip:
        """
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
