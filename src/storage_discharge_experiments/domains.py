from mpl_toolkits.basemap import Basemap
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np


def get_domain_coords_and_basemap(coord_file = "~/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260/Samples/quebec_220x220_198505/pm1985050100_00000000p",
                                  lon1 = 180, lat1 = 0, lon2 = 180, lat2 = 0
                                  ):
    rpnObj = RPN(coord_file)
    lons2d, lats2d = rpnObj.get_longitudes_and_latitudes()

    basemap = Basemap(projection="omerc")
    pass


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  