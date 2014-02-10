from mpl_toolkits.basemap import Basemap

__author__ = 'huziy'

import numpy as np

class GridConfig():
    #params
    dx = 0.1
    dy = 0.1
    iref = 142  # no need to do -1, doing it later in the formulas
    jref = 122
    xref = 180 #rotated longitude
    yref = 0   #rotated latitude

    #projection parameters
    lon1 = -68.0
    lat1 = 52.0

    lon2 = 16.65
    lat2 = 0.0

    ni = 260
    nj = 260

    projection = "rotpole"



    def __init__(self):
        pass


    @classmethod
    def get_default_for_resolution(cls, res = 0.5):
        """
        :rtype GridConfig
        """
        obj = GridConfig()
        if res == 0.5:
            obj.dx = obj.dy = 0.5
            obj.iref = 46 #starts from 1 not 0!!
            obj.jref = 42  #starts from 1 not 0!!
            obj.ni = 86
            obj.nj = 86
        elif res == 0.1:
            pass

        return obj


    def get_basemap(self):
        return Basemap()



def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  