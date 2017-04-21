
import xarray as xr

import numpy as np


import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import Basemap

from util import plot_utils


def calculate_composits():
    pass


def get_mean_for_years(darr, year_list=None):
    print(darr.time)

    # +1 because November and December are from the previous year
    sel_vec = np.where([(t.year - 1 in year_list) for t in pd.to_datetime(darr.time.values)])[0]
    res = darr[sel_vec, :, :].mean(dim="time")
    print(darr.time[sel_vec])
    return res


def get_composit_for_name(ds, vname, high_years_list, low_years_list):
    high = get_mean_for_years(ds[vname], year_list=high_years_list)
    low = get_mean_for_years(ds[vname], year_list=low_years_list)
    return high - low




def main():

    plot_utils.apply_plot_params(width_cm=20, height_cm=20, font_size=10)

    high_hles_years = [1993, 1995, 1998]
    low_hles_years = [1997, 2001, 2006]
    data_path = "/BIG1/skynet1_rech1/diro/sample_obsdata/eraint/eraint_uvslp_years_198111_201102_NDJmean_ts.nc"





    with xr.open_dataset(data_path) as ds:
        print(ds)


        u = get_composit_for_name(ds, "u10", high_years_list=high_hles_years, low_years_list=low_hles_years)
        v = get_composit_for_name(ds, "v10", high_years_list=high_hles_years, low_years_list=low_hles_years)
        msl = get_composit_for_name(ds, "msl", high_years_list=high_hles_years, low_years_list=low_hles_years)

        lons = ds["longitude"].values
        lats = ds["latitude"].values

        print(lats)
        print(msl.shape)
        print(lons.shape, lats.shape)


        lons2d, lats2d = np.meshgrid(lons, lats)


    fig = plt.figure()

    map = Basemap(llcrnrlon=-130, llcrnrlat=22, urcrnrlon=-28,
                  urcrnrlat=65, projection='lcc', lat_1=33, lat_2=45,
                  lon_0=-95, resolution='i', area_thresh=10000)


    clevs = np.arange(-11.5, 12, 1)
    cmap = cm.get_cmap("bwr", len(clevs) - 1)
    bn = BoundaryNorm(clevs, len(clevs) - 1)

    x, y = map(lons2d, lats2d)
    im = map.contourf(x, y, msl / 100, levels=clevs, norm=bn, cmap=cmap) # convert to mb (i.e hpa)
    map.colorbar(im)


    stride = 2
    ux, vy = map.rotate_vector(u, v, lons2d, lats2d)
    qk = map.quiver(x[::stride, ::stride], y[::stride, ::stride], ux[::stride, ::stride], vy[::stride, ::stride],
               scale=10, width=0.01, units="inches")
    plt.quiverkey(qk, 0.5, -0.1, 2, "2 m/s", coordinates="axes")


    map.drawcoastlines(linewidth=0.5)
    map.drawcountries()
    map.drawstates()
    #plt.show()

    fig.savefig("hles_wind_compoosits.png", bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    main()