from mpl_toolkits.basemap import Basemap
from domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt



def demo_north_pole():

    r = RPN(path = "/home/huziy/skynet3_rech1/classOff_Andrey/era2/temp_3d")
    t = r.get_first_record_for_name("I0")
    lon, lat = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    r.close()
    nx, ny = lon.shape

    lon_0, lat_0 = lon[nx//2, ny//2], lat[nx//2, ny//2]


    basemap = Basemap(projection = "omerc", lon_1=60, lat_1 = 89.999, lon_2=-30, lat_2=0, no_rot=True,
        lon_0 = lon_0, lat_0 = lat_0,
        llcrnrlon=lon[0, 0], llcrnrlat=lat[0,0],
        urcrnrlon=lon[-1, -1], urcrnrlat=lat[-1, -1]

    )

    x, y = basemap(lon, lat)


    basemap.contourf(x, y, t)
    basemap.drawcoastlines()
    basemap.colorbar()

    #basemap.shadedrelief()
    plt.show()

def demo_arctic_proj():
    rll = RotatedLatLon(lon1=60, lat1=90, lon2=-30, lat2=0)

    print(rll.get_north_pole_coords(), rll.get_true_pole_coords_in_rotated_system())

    pass




def main():
    #TODO: implement


    #basemap = Basemap(projection = "npstere", boundinglat=60, lon_0=-100, round=True)
    #basemap.shadedrelief()
    #plt.show()
    demo_arctic_proj()

    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    #demo_north_pole()
    print("Hello world")
  