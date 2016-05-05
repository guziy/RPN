from mpl_toolkits.basemap import Basemap

__author__ = 'huziy'

import numpy as np

USE_SKIMAGE = True
try:
    from skimage.util import view_as_blocks
except ImportError as ie:
    view_as_blocks = None
    USE_SKIMAGE = False

DEFAULT_BG_COLOR = "0.6"


class BasemapInfo(object):
    """
    :type basemap: Basemap
    """
    # Object containing info for plotting on a map
    def __init__(self, lons=None, lats=None, bmp=None):
        self.lons = lons.copy()
        self.lons[self.lons > 180] -= 360
        self.lats = lats
        self.basemap = bmp

        self.map_bg_color = DEFAULT_BG_COLOR
        self.should_draw_grey_map_background = False
        self.should_draw_basin_boundaries = True

    def get_proj_xy(self):
        return self.basemap(self.lons, self.lats)


    def basemap_for_extended_region(self, marginx=10, marginy=10):
        xx, yy = self.get_proj_xy()
        xx1, yy1 = expand_domain(xx, yy, marginx=marginx, marginy=marginy)


        import copy

        proj_params = copy.copy(self.basemap.projparams)

        del proj_params["proj"]
        del proj_params["R"]
        del proj_params["units"]
        del proj_params["o_proj"]

        proj_params.update({
            "llcrnrx": xx1[0, 0],
            "llcrnry": yy1[0, 0],
            "urcrnrx": xx1[-1, -1],
            "urcrnry": yy1[-1, -1],
            "projection": self.basemap.projection
        })

        b1 = Basemap(**proj_params)

        b1.llcrnrx = xx1[0, 0]
        b1.llcrnry = yy1[0, 0]
        b1.urcrnrx = xx1[-1, -1]
        b1.urcrnry = yy1[-1, -1]
        return b1

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

    def draw_map_background(self, ax):
        """

        :param ax: Axis
        """
        if self.should_draw_grey_map_background:
            self.basemap.drawmapboundary(ax=ax, fill_color=self.map_bg_color)


def get_gridsize(xx, axis=0):
    """
    infer, approximately, the grid size from the given 2d field of a coordinate
    :param xx:
    """
    return (xx[-1, -1] - xx[0, 0]) / (xx.shape[axis] - 1)


def expand_domain(xx, yy, marginx=10, marginy=10):
    dx = get_gridsize(xx, axis=0)
    dy = get_gridsize(yy, axis=1)

    xx1 = np.zeros((2, 2))
    yy1 = np.zeros((2, 2))

    xx1[0, :] = xx[0, 0] - marginx * dx
    xx1[-1, :] = xx[-1, -1] + marginx * dx

    yy1[0, :] = yy[0, 0] - marginy * dy
    yy1[-1, :] = yy[-1, -1] + marginy * dy

    return xx1, yy1