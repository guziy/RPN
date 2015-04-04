from mpl_toolkits.basemap import Basemap
from domains.rotated_lat_lon import RotatedLatLon

__author__ = 'huziy'

import numpy as np


class GridConfig():
    projection = "rotpole"

    def __init__(self, **kwargs):
        self.dx = self.dy = kwargs["dx"]
        self.iref, self.jref = kwargs["iref"], kwargs["jref"]
        self.ni, self.nj = kwargs["ni"], kwargs["nj"]
        self.xref, self.yref = kwargs["xref"], kwargs["yref"]
        self.lon1, self.lat1 = kwargs["lon1"], kwargs["lat1"]
        self.lon2, self.lat2 = kwargs["lon2"], kwargs["lat2"]
        self.ni, self.nj = kwargs["ni"], kwargs["nj"]


    @classmethod
    def get_default_for_resolution(cls, res=0.5):
        """
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


    def get_basemap(self):
        return Basemap()


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
    pass


if __name__ == "__main__":
    main()
    print("Hello world")
  