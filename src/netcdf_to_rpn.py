
__author__="huziy"
__date__ ="$Aug 20, 2011 1:45:02 PM$"

import netCDF4 as nc
from rpn import RPN

def convert(nc_path = 'data/directions_qc_dx0.1.nc'):
    ds = nc.Dataset(nc_path)

    ncNameToRpnName = {'flow_direction_value': 'fldr', 'slope': 'slop', 
                        'channel_length':'leng', 'accumulation_area':'facc',
                        'lon': '>>', 'lat' : '^^'
                      }
    rObj = RPN('infocell.rpn' , mode = 'w')
    for ncName, rpnName in ncNameToRpnName.iteritems():
        data = ds.variables[ncName][:]
        grid_type = 'E' if rpnName in ['>>', '^^'] else 'Z'
        rObj.write_2D_field(name = rpnName, level = 1, data = data, grid_type = grid_type)
    rObj.close()


    print ds.variables.keys()

    pass


import application_properties
if __name__ == "__main__":
    application_properties.set_current_directory()
    convert()
    print "Hello World"
