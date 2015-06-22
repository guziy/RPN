__author__ = 'huziy'

import numpy as np

USE_SKIMAGE = True
try:
    from skimage.util import view_as_blocks
except ImportError as ie:
    USE_SKIMAGE = False


class BasemapInfo(object):
    # Object containing info for plotting on a map
    def __init__(self, lons=None, lats=None, bmp=None):
        self.lons = lons.copy()
        self.lons[self.lons > 180] -= 360
        self.lats = lats
        self.basemap = bmp

    def get_proj_xy(self):
        return self.basemap(self.lons, self.lats)

    def get_aggregated(self, nagg_x=2, nagg_y=2):

        if USE_SKIMAGE:
            new_lons = view_as_blocks(self.lons, (nagg_x, nagg_y)).mean(axis=2).mean(axis=2)
            new_lats = view_as_blocks(self.lats, (nagg_x, nagg_y)).mean(axis=2).mean(axis=2)
        else:
            nx, ny = self.lons.shape
            new_lons, new_lats = np.zeros((nx // nagg_x, ny // nagg_y)), np.zeros((nx // nagg_x, ny // nagg_y))

            for i in range(0, nx, nagg_x):
                for j in range(0, ny, nagg_y):
                    i1 = i // nagg_x
                    j1 = j // nagg_y

                    new_lons[i1, j1] = np.mean(self.lons[i:i + nagg_x, j:j + nagg_y])
                    new_lats[i1, j1] = np.mean(self.lons[i:i + nagg_x, j:j + nagg_y])

        return BasemapInfo(lons=new_lons, lats=new_lats, bmp=self.basemap)