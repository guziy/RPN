from datetime import datetime
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from rpn.domains.rotated_lat_lon import RotatedLatLon
from application_properties import main_decorator
from mpl_toolkits.basemap import cm as cm_basemap, Basemap

__author__ = 'huziy'

from rpn.rpn import RPN

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

"""
Read data from an RPN file, calculate daily mean fields and plot them
"""


@main_decorator
def main():
    path = "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/Global_NA_v1/Samples/Global_NA_v1_201306/pm2013010100_00016320p"
    varname = "PR"
    plot_units = "mm/day"
    mult_coeff = 1000 * 24 * 3600
    add_offset = 0

    img_folder = "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/glob_sim/{}/daily/{}".format(varname, os.path.basename(path))
    img_folder = Path(img_folder)
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    r = RPN(path=path)

    pr = r.get_all_time_records_for_name(varname=varname)

    lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    rll = RotatedLatLon(**r.get_proj_parameters_for_the_last_read_rec())



    bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons2d,
                                               lats2d=lats2d,
                                               resolution="c", no_rot=True)

#    bmp = Basemap(projection="robin", lon_0=180)

    xx, yy = bmp(lons2d, lats2d)


    dates = list(sorted(pr.keys()))
    data = np.array([pr[d] for d in dates])

    p = pd.Panel(data=data, items=dates, major_axis=range(data.shape[1]), minor_axis=range(data.shape[2]))

    p_daily = p.groupby(lambda d: d.day, axis="items").mean()



    clevs = [0, 0.1, 1, 1.5, 2, 5, 10, 20, 40, 60, 80]
    norm = BoundaryNorm(clevs, ncolors=len(clevs) - 1)
    cmap = cm.get_cmap(cm_basemap.s3pcpn, len(clevs) - 1)
    for k in p_daily:

        field = p_daily[k].values * mult_coeff + add_offset
        fig = plt.figure()
        plt.title("{}, {}/{}/{}, {}".format(varname, k, dates[0].month, dates[0].year, plot_units))


        # cs = bmp.contourf(xx, yy, field, clevs, norm=norm, extend="max", cmap=cmap)
        cs = bmp.pcolormesh(xx, yy, field, norm=norm, cmap=cmap)
        bmp.colorbar(cs, ticks=clevs, extend="max")
        img_file = img_folder.joinpath("{:02d}-{:02d}-{}.png".format(k, dates[0].month, dates[0].year))
        bmp.drawcoastlines()
        bmp.drawstates()
        # bmp.drawcounties()
        bmp.drawcountries()

        plt.savefig(img_file.open("wb"))
        plt.close(fig)



    print(pr[dates[0]].mean())



if __name__ == '__main__':
    main()
