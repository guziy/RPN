from permafrost.active_layer_thickness import CRCMDataManager

__author__ = 'huziy'

import numpy as np
from cru.temperature import CRUDataManager

import draw_regions

def main():
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()

    cru = CRUDataManager()
    cru_data = cru.get_mean(1981,2009, months = [6,7,8])
    cru_data_interp = cru.interpolate_data_to(cru_data, lons2d, lats2d)


    crcm = CRCMDataManager()
    #TODO: find out what is the variable name
    crcm_data = crcm.get_mean_over_months_of_2d_var(1981,2009, months = [6,7,8], var_name="")



    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  