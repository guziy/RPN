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
import cartopy.crs as ccrs

import os

"""
Read data from an RPN file, calculate daily mean fields and plot them
"""


@main_decorator
def main():
    path = "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/Global_NA_v1/Samples/Global_NA_v1_201306/pm2013010100_00017280p"
    varname = "PR"
    plot_units = "mm/day"
    mult_coeff = 1000 * 24 * 3600
    add_offset = 0

    img_folder = "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/glob_sim/{}/monthly/{}".format(varname, os.path.basename(path))
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

    # p_daily = p.groupby(lambda d: d.day, axis="items").mean()
    p_daily = p.apply(np.mean, axis="items")

    print(p_daily.head())

    lons2d[lons2d > 180] -= 360

    bmap_params = bmp.projparams
    bmap_params.update({
        'llcrnrlon': lons2d[0, 0], 'urcrnrlon': lons2d[-1, -1], 'llcrnrlat': lats2d[0, 0], 'urcrnrlat': lats2d[-1, -1]
    })
    rpole_crs = ccrs.RotatedPole(pole_longitude=bmap_params["lon_0"] + 180,
                                 pole_latitude=bmap_params["o_lat_p"])



    clevs = [0, 0.01, 0.1, 1, 1.5, 2, 5, 10, 20, 40, 60, 80]
    norm = BoundaryNorm(clevs, ncolors=len(clevs) - 1)
    cmap = cm.get_cmap(cm_basemap.s3pcpn, len(clevs) - 1)
    field = p_daily.values * mult_coeff + add_offset
    fig = plt.figure()
    plt.title("{}, {}/{}/{}, {}".format(varname, 1, dates[0].month, dates[0].year, plot_units))




    ax = plt.axes(projection=rpole_crs)
    ax.coastlines(resolution='110m')

    ax.gridlines()
    ax.gridlines()


    # cs = bmp.contourf(xx, yy, field, clevs, norm=norm, extend="max", cmap=cmap)
    cs = ax.pcolormesh(lons2d[:-1, :-1], lats2d[:-1, :-1], field[:-1, :-1], norm=norm, cmap=cmap, transform=rpole_crs)
    plt.colorbar(cs, ticks=clevs, extend="max", ax=ax)
    img_file = img_folder.joinpath("{:02d}-{:02d}-{}.png".format(1, dates[0].month, dates[0].year))
    # bmp.drawcoastlines()
    # bmp.drawstates()
    # bmp.drawcounties()
    # bmp.drawcountries()

    plt.savefig(img_file.open("wb"))
    plt.close(fig)
    print(pr[dates[0]].mean())



if __name__ == '__main__':
    main()
