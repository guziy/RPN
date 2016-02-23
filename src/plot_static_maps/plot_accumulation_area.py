from netCDF4 import Dataset
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap, maskoceans

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt


def get_nps_basemap(lons, lats, slope):
    return Basemap(projection="npstere", boundinglat=lats[slope >= 0].min(), lon_0=-115)


def get_omerc_basemap_africa1(lons, lats, lon1=0, lat1=0.00001, lon2=89.99999, lat2=0.000001):
    return Basemap(projection="omerc",
                   lon_1=lon1, lon_2=lon2, lat_1=lat1, lat_2=lat2,
                   llcrnrlon=lons[0, 0], llcrnrlat=lats[0, 0],
                   urcrnrlon=lons[-1, -1], urcrnrlat=lats[-1, -1], no_rot=True
                   )
    pass


def get_omerc_basemap_quebec(lons, lats, lon1=-68, lat1=52, lon2=16.65, lat2=0):
    return Basemap(projection="omerc",
                   lon_1=lon1, lon_2=lon2, lat_1=lat1, lat_2=lat2,
                   llcrnrlon=lons[0, 0], llcrnrlat=lats[0, 0],
                   urcrnrlon=lons[-1, -1], urcrnrlat=lats[-1, -1], no_rot=True, resolution="l"
                   )


def main():
    AFRIC = 1
    QUEBEC = 2

    region = AFRIC

    if region == QUEBEC:
        data_path = "/home/huziy/skynet3_rech1/Netbeans Projects/Java/DDM/directions_with_drainage_density/directions_qc_dx0.1deg_5.nc"
        out_path = "qc_acc_area_0.1deg.pdf"
    elif region == AFRIC:
        data_path = "/home/huziy/skynet3_rech1/Netbeans Projects/Java/DDM/directions_africa_dx0.44deg.v3.nc"
        out_path = "af_acc_area_0.44deg.pdf"
    else:
        raise Exception("Unknown region...")

    #
    varname = "accumulation_area"
    ds = Dataset(data_path)

    data = ds.variables[varname][30:-30, 30:-30]

    lons = ds.variables["lon"][30:-30, 30:-30]
    lats = ds.variables["lat"][30:-30, 30:-30]
    slope = ds.variables["slope"][30:-30, 30:-30]

    fig = plt.figure()
    print(data.min(), data.max())
    ax = plt.gca()

    data = np.ma.masked_where(slope < 0, data)

    basemap = get_omerc_basemap_africa1(lons, lats)

    lons[lons > 180] -= 360
    x, y = basemap(lons, lats)

    data = maskoceans(lons, lats, data, inlands=False)

    img = basemap.contourf(x, y, np.ma.log(data), cmap=cm.get_cmap("winter_r"))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img, cax=cax)
    cax.set_title("$\ln\left({\\rm km^2 }\\right)$ \n")

    basemap.drawcoastlines(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path)


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    main()
    print("Hello world")
