
__author__="huziy"
__date__ ="$Aug 20, 2011 1:45:02 PM$"

import netCDF4 as nc

def convert(nc_path = 'data/directions_qc_dx0.1.nc'):
    ds = nc.Dataset(nc_path)

    print ds.variables.keys()

    pass


import application_properties
if __name__ == "__main__":
    application_properties.set_current_directory()
    convert()
    print "Hello World"
