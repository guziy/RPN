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

    def get_basemap(self, lons, lats, **kwargs):
        return self.get_rot_latlon_proj_obj().get_basemap_object_for_lons_lats(lons2d=lons,
                                                                               lats2d=lats,
                                                                               **kwargs)

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
            subgr.ni = self.ni

        if dj > 0:
            subgr.nj = dj
        else:
            subgr.nj = self.nj


        subgr.iref -= i0
        subgr.jref -= j0

        return subgr



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
  