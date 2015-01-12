__author__ = 'san'

from rpn.rpn import RPN
from rpn.domains.rotated_latlon import RotatedLatLon
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap




def get_data_and_coords():
    """
    :rtype: dict
    :return: dict with {level: field} data
    """
    path = "/skynet3_rech1/huziy/geofields_interflow_exp/pm1979010100_00000000p"
    vname = "D9"

    r = RPN(path)
    data = r.get_4d_field(name=vname)

    params = r.get_proj_parameters_of_the_last_read_rec()
    lons, lats = r.get_longitudes_and_latitude_of_the_last_read_rec()

    rll = RotatedLatLon(**params)
    bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)

    r.close()
    return data, lons, lats, bmp




def main():

    level_to_plot = 1
    data, lons, lats, bmp = get_data_and_coords()
    print data.keys()

    fig = plt.figure()

    #TODO: finish plotting


    fig.savefig("thfc_lev_{}.png".format(level_to_plot))



if __name__ == '__main__':
    main()