from permafrost.active_layer_thickness import CRCMDataManager

__author__ = 'huziy'

import numpy as np

def main():
    crcm = CRCMDataManager(data_folder = "data/CORDEX")
    snow_mass = crcm.get_mean_over_months_of_2d_var(1981, 1997, months=[12,1,2], var_name="I5")
    swe_model = snow_mass / 1000.0





    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  