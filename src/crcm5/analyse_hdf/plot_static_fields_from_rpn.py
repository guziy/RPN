__author__ = 'huziy'


import plot_static_fields as psf
from rpn.rpn import RPN
import numpy as np
import pandas as pd
from domains.rotated_lat_lon import RotatedLatLon
import matplotlib.pyplot as plt

def main(path = "/skynet3_rech1/huziy/geof_lake_infl_exp/geophys_Quebec_0.1deg_260x260_with_dd_v6_with_ITFS"):
    r = RPN(path)

    varnames = ["ITFS"]


    ncols = 3
    nrows = len(varnames) // 3

    fig = plt.figure()
    varname_to_field = {}
    for vname in varnames:

        data = r.get_first_record_for_name(vname)
        varname_to_field[vname] = data
        data = np.ma.masked_where(data < 0, data)
        lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()
        params = r.get_proj_parameters_for_the_last_read_rec()
        print params
        rll = RotatedLatLon(**params)
        b = rll.get_basemap_object_for_lons_lats(lons2d, lats2d)
        x, y = b(lons2d, lats2d)
        b.drawcoastlines()
        img = b.pcolormesh(x, y, data)
        b.colorbar()



    fig = plt.figure()
    itfs = varname_to_field["ITFS"]
    plt.hist(itfs[itfs >= 0], bins = 100)


    plt.show()

    r.close()
    pass


if __name__ == "__main__":
    main()