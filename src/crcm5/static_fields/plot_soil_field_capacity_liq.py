import os
from matplotlib import cm

__author__ = 'san'

from rpn.rpn import RPN
from rpn.domains.rotated_lat_lon import RotatedLatLon
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, maskoceans
import numpy as np


def get_data_and_coords():
    """
    :rtype: (dict, np.ndarray, np.ndarray, Basemap)
    :return: dict with {level: field} data
    """
    path = "/skynet3_rech1/huziy/geofields_interflow_exp/pm1979010100_00000000p"
    vname = "D9"

    r = RPN(path)
    data = list(r.get_4d_field(name=vname).items())[0][1]

    params = r.get_proj_parameters_for_the_last_read_rec()
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

    rll = RotatedLatLon(**params)
    bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)

    r.close()
    return data, lons, lats, bmp




def main():

    # create the image folder if necessary
    img_folder = "bulk_field_capacity_model"
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    data, lons, lats, bmp = get_data_and_coords()
    lons[lons > 180] -= 360
    print(list(data.keys()))

    # reproject coords
    x, y = bmp(lons, lats)

    clevs = np.arange(0, 0.5, 0.02)
    cmap = cm.get_cmap("rainbow", lut=len(clevs))
    # plot for all levels right away
    for lev, field in data.items():
        fig = plt.figure()
        plt.title(r"$\theta_{\rm fc}$, " + "soil lev = {}".format(lev))
        to_plot = maskoceans(lons, lats, field, inlands=True)
        img = bmp.contourf(x, y, to_plot, levels=clevs, cmap=cmap)
        bmp.colorbar(img)

        bmp.drawcoastlines()

        print("lev={}, fc-min={}, fc-max={}".format(lev, field.min(), field.max()))

        fname = "thfc_lev_{}.png".format(lev)
        fig.savefig(os.path.join(img_folder, fname))
        plt.close(fig)



if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()