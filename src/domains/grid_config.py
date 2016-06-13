from mpl_toolkits.basemap import Basemap
from domains.rotated_lat_lon import RotatedLatLon

__author__ = 'huziy'

import numpy as np


class GridConfig(object):
    projection = "rotpole"

    def __init__(self, **kwargs):
        self.dx = self.dy = kwargs.get("dx", -1)
        self.iref, self.jref = kwargs.get("iref", -1), kwargs.get("jref", -1)

        self.xref, self.yref = kwargs.get("xref", -1), kwargs.get("yref", -1)
        self.ni, self.nj = kwargs.get("ni", -1), kwargs.get("nj", -1)

        # interpolated driving data (width of the outer band in number of gridpoints)
        self.halo = 10

        # size of the blending zone in grid points
        self.blendig = 10


        self.rll = None
        if "rll" not in kwargs:
            self.lon1, self.lat1 = kwargs.get("lon1"), kwargs.get("lat1")
            self.lon2, self.lat2 = kwargs.get("lon2"), kwargs.get("lat2")
            if None not in (self.lon1, self.lon2, self.lat1, self.lat2):
                self.rll = RotatedLatLon(lon1=self.lon1, lon2=self.lon2, lat1=self.lat1, lat2=self.lat2)
        else:
            self.rll = kwargs.get("rll")

    @classmethod
    def get_default_for_resolution(cls, res=0.5):
        """
        :param res:
        :rtype GridConfig
        """
        obj = GridConfig()
        obj.dx = obj.dy = res
        if res == 0.5:
            obj.iref = 46  # starts from 1 not 0!!
            obj.jref = 42  # starts from 1 not 0!!
            obj.ni = 86
            obj.nj = 86
        elif res == 0.1:
            obj.iref = 142  # no need to do -1, doing it later in the formulas
            obj.jref = 122
            obj.xref = 180  # rotated longitude
            obj.yref = 0  # rotated latitude

            # projection parameters
            obj.lon1 = -68.0
            obj.lat1 = 52.0

            obj.lon2 = 16.65
            obj.lat2 = 0.0

            obj.ni = 260
            obj.nj = 260

        return obj

    def get_basemap(self, lons=None, lats=None, **kwargs):

        if lons is None:

            lonr = [(i - (self.iref - 1)) * self.dx + self.xref for i in range(self.ni)]
            latr = [(j - (self.jref - 1)) * self.dy + self.yref for j in range(self.nj)]

            latr, lonr = np.meshgrid(latr, lonr)

            lons = np.zeros((2, 2))
            lats = np.zeros((2, 2))

            for i in [-1, 0]:
                for j in [-1, 0]:
                    shiftx = self.dx / 2.0
                    shifty = self.dy / 2.0

                    shiftx = -shiftx if i == 0 else shiftx
                    shifty = -shifty if i == 0 else shifty


                    lons[i, j], lats[i, j] = self.rll.toGeographicLonLat(lonr[i, j] + shiftx, latr[i, j] + shifty)

        return self.get_rot_latlon_proj_obj().get_basemap_object_for_lons_lats(lons2d=lons,
                                                                               lats2d=lats,
                                                                               **kwargs)




    def get_free_zone_corners(self, halo=10, blending=10):
        lonr = [(i - (self.iref - 1)) * self.dx + self.xref for i in range(self.ni)]
        latr = [(j - (self.jref - 1)) * self.dy + self.yref for j in range(self.nj)]

        latr, lonr = np.meshgrid(latr, lonr)

        lons = np.zeros((2, 2))
        lats = np.zeros((2, 2))

        margin = halo + blending

        for i in [-margin - 1, margin]:
            mulx = -1 if i >= 0 else 1
            i1 = 0 if i > 0 else -1
            shiftx = mulx * self.dx / 2.0

            for j in [-margin - 1, margin]:
                j1 = 0 if j > 0 else -1
                muly = -1 if j >= 0 else 1
                shifty = muly * self.dy / 2.0


                lons[i1, j1], lats[i1, j1] = self.rll.toGeographicLonLat(lonr[i, j] + shiftx, latr[i, j] + shifty)

        return lons, lats




    def get_rot_latlon_proj_obj(self):
        return self.rll



    def subgrid(self, i0, j0, di=-1, dj=-1):

        """

        :param i0: 0-based i-index of the lower left corner of the domain
        :param j0:
        :param di: number of grid points in i direction
        :param dj: number of grid points in j direction
        """


        subgr = GridConfig(rll=self.rll, dx=self.dx, dy=self.dy, xref=self.xref, yref=self.yref)

        if di > 0:
            subgr.ni = di
        else:
            subgr.ni = self.ni - i0

        if dj > 0:
            subgr.nj = dj
        else:
            subgr.nj = self.nj - j0


        subgr.iref = self.iref - i0
        subgr.jref = self.jref - j0

        return subgr



    def copy(self):
        return self.subgrid(0, 0, di=self.ni, dj=self.nj)



    def double_resolution(self):
        gc = GridConfig(rll=self.rll, dx=self.dx / 2.0, dy=self.dy / 2.0, xref=self.xref, yref=self.yref)
        gc.ni = 2 * self.ni
        gc.nj = 2 * self.nj

        gc.iref = 2 * self.iref
        gc.jref = 2 * self.jref
        return gc

    def double_resolution_keep_free_domain_same(self, halo_pts=10, blending_pts=10):
        gc = GridConfig(rll=self.rll, dx=self.dx / 2.0, dy=self.dy / 2.0, xref=self.xref, yref=self.yref)
        margin_pts = halo_pts + blending_pts
        gc.ni = 2 * (self.ni - 2 * margin_pts) + 2 * margin_pts
        gc.nj = 2 * (self.nj - 2 * margin_pts) + 2 * margin_pts

        gc.iref = 2 * (self.iref - margin_pts) + margin_pts
        gc.jref = 2 * (self.jref - margin_pts) + margin_pts
        return gc

    def decrease_resolution_keep_free_domain_same(self, factor, halo_pts=10, blending_pts=10):
        gc = GridConfig(rll=self.rll, dx=self.dx * factor, dy=self.dy * factor, xref=self.xref, yref=self.yref)
        margin_pts = halo_pts + blending_pts
        gc.ni = (self.ni - 2 * margin_pts) / factor + 2 * margin_pts
        gc.nj = (self.nj - 2 * margin_pts) / factor + 2 * margin_pts


        # Change the reference point if the new iref and jref cannot be the same

        new_iref = self.iref - margin_pts
        new_jref = self.jref - margin_pts


        new_iref = new_iref // factor + (new_iref % factor != 0)
        x00 = self.xref + self.dx * (margin_pts + 1 - self.iref) - self.dx / 2.0
        new_xref = x00 + new_iref * self.dx * factor - self.dx * factor / 2.0

        new_jref = new_jref // factor + (new_jref % factor != 0)
        y00 = self.yref + self.dy * (margin_pts + 1 - self.jref) - self.dy / 2.0
        new_yref = y00 + new_jref * self.dy * factor - self.dy * factor / 2.0


        gc.iref = new_iref + margin_pts
        gc.jref = new_jref + margin_pts

        gc.xref = new_xref
        gc.yref = new_yref


        gc.ni = int(gc.ni)
        gc.nj = int(gc.nj)


        return gc

    def move(self, di=0, dj=0):
        gc = GridConfig(rll=self.rll, dx=self.dx, dy=self.dy, xref=self.xref, yref=self.yref, ni=self.ni, nj=self.nj)
        gc.iref -= di
        gc.jref -= dj
        return gc

    def expand(self, di=0, dj=0):
        gc = GridConfig(rll=self.rll, dx=self.dx, dy=self.dy, xref=self.xref, yref=self.yref, ni=self.ni + di, nj=self.nj + dj)
        gc.iref = self.iref
        gc.jref = self.jref
        return gc



    def __str__(self):
        s = """
              Grd_ni        =  {ni}     , Grd_nj         =    {nj}     ,
              Grd_dx        =  {dx}     , Grd_dy         =    {dy},
              Grd_iref      =  {iref}     ,  Grd_jref       =   {jref}     ,
              Grd_latr      =    {latref}   ,  Grd_lonr       =  {lonref}   ,
              Grd_xlat1     =   {lat1}   ,  Grd_xlon1       = {lon1}    ,
              Grd_xlat2     =    {lat2}    ,  Grd_xlon2       =  {lon2}    ,
        """.format(ni=self.ni, nj=self.nj, dx=self.dx, dy=self.dy, iref=self.iref, jref=self.jref,
                   latref=self.yref, lonref=self.xref, lat1=self.rll.lat1, lon1=self.rll.lon1,
                   lat2=self.rll.lat2, lon2=self.rll.lon2)

        return s


def get_rotpole_for_na_glaciers():
    """
    Glacier grid
      Grd_typ_S     = 'LU'     ,
      Grd_ni        =  196     ,  Grd_nj          =  140     ,
      Grd_dx        =    0.1375,  Grd_dy          =    0.1375,
      Grd_iref      =  106     ,  Grd_jref        =   70     ,
      Grd_latr      =    0.0   ,  Grd_lonr        =  180.0   ,
      Grd_xlat1     =   57.5   ,  Grd_xlon1       = -130.    ,
      Grd_xlat2     =    0.    ,  Grd_xlon2       =  -40.    ,
    :return:
    """
    params = dict(
        lon1=-130, lat1=57.5,
        lon2=-40.0, lat2=0.0
    )
    return RotatedLatLon(**params)


def main():
    GridConfig.get_default_for_resolution(0.1)


if __name__ == "__main__":
    main()
    print("Hello world")
  