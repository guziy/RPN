from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator
from matplotlib.transforms import Bbox, Affine2D
from offline_route import plot_seasonal_means
from offline_route import plot_changes_in_seasonal_means
import matplotlib.pyplot as plt
import numpy as np

import mpl_toolkits.axisartist.floating_axes as floating_axes


__author__ = 'huziy'

from netCDF4 import Dataset

def get_accumulation_area(path="/skynet1_rech3/huziy/offline_stfl/canesm/discharge_1958_01_01_00_00.nc"):
    """
    Reads the accumulation area field in km**2
    :param path:
    """
    ds = Dataset(path)
    acc = ds.variables["accumulation_area"][:]
    lons, lats = ds.variables["longitude"][:], ds.variables["latitude"][:]
    ds.close()
    return lons, lats, acc

def plot_usual_acc_area(bmap, x, y, data):
    fig = plt.figure(figsize=(10, 6))
#    bmap.etopo()
#    bmap.warpimage()
    ax = plt.gca()

    #img = bmap.pcolormesh(x, y, data, norm=bn, cmap=cm.get_cmap("jet", len(boundaries) - 1))
    img = bmap.contourf(x, y, data, norm=LogNorm())
    cb = bmap.colorbar(img)
    cb.ax.set_title(r"${\rm km^2}$")
    bmap.drawcoastlines(linewidth=0.3)
    bmap.readshapefile("data/shp/wri_basins2/wribasin", "basin", color="k", linewidth=2, ax=ax)
#    bmap.drawrivers(color="blue")
#    bmap.drawmeridians(np.arange(-180, 180, 20))
#    bmap.drawparallels(np.arange(-90, 110, 20))

    fp = FontProperties(weight="bold", size=6)
    ax.annotate("Mackenzie", (0.2, 0.7), xycoords="axes fraction", font_properties=fp)
    ax.annotate("Ob", (0.8, 0.3), xycoords="axes fraction", font_properties=fp)
    ax.annotate("Yenisei", (0.8, 0.48), xycoords="axes fraction", font_properties=fp)
    ax.annotate("Lena", (0.8, 0.63), xycoords="axes fraction", font_properties=fp)

    fig.savefig("simple_labels_white_ocean_no_meridians_no_parallels_solid.png")
    plt.show()







def plot_default_topo(bmap,x, y, data):
    assert isinstance(bmap, Basemap)
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(1, 2, width_ratios=[4, 1], height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0, 1])

    bmap.etopo()
    img = bmap.contourf(x, y, data, norm=LogNorm(), alpha=0.8, ax=ax)
    cb = bmap.colorbar(img)
    cb.ax.set_title(r"${\rm km^2}$")
    bmap.drawcoastlines(linewidth=0.3)

    bmap.drawmeridians(np.arange(-180, 180, 20))
    bmap.drawparallels(np.arange(-90, 110, 20))
    bmap.readshapefile("data/shp/wri_basins2/wribasin", "basin", color="k", linewidth=2, ax=ax)



    lower_left_lat_lon = (-50, 45)
    axins = fig.add_subplot(gs[0, 0])  # plt.axes([-0.15, 0.1, 0.6, 0.8])  # zoomed_inset_axes(ax, 2, loc=1)  # zoom = 6


    #bmap.etopo(ax=axins)

    bm_zoom = plot_changes_in_seasonal_means.get_arctic_basemap_nps(round=False, resolution="l")
    lower_left_xy = bmap(*lower_left_lat_lon)
    upper_right_xy = bmap(-160, 55)

    bm_zoom.etopo(ax=axins)
    #axins.set_xlim(lower_left_xy[0], lower_left_xy[0] + 400000)
    #axins.set_ylim(lower_left_xy[1], lower_left_xy[1] + 5000000)

    print(lower_left_xy)
    print(upper_right_xy)


    #bm_zoom.etopo(ax=axins)
    assert isinstance(axins, Axes)

    img = bm_zoom.contourf(x, y, data, norm=LogNorm(), alpha=0.8, ax=axins)
    bm_zoom.drawcoastlines(linewidth=0.3)
    bm_zoom.readshapefile("data/shp/wri_basins2/wribasin", "basin", color="k", linewidth=2, ax=axins)


    axins.set_xlim(lower_left_xy[0], upper_right_xy[0])
    axins.set_ylim(lower_left_xy[1], upper_right_xy[1])




    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
    fig.tight_layout()
    fig.savefig("simple2.png", bbox_inches="tight")

    plt.show()


def main():
    bmap = plot_seasonal_means.get_arctic_basemap_nps(resolution="l")
    land_sea_mask = plot_changes_in_seasonal_means.get_land_sea_glaciers_mask_from_geophysics_file(
        path="/skynet1_rech3/huziy/geophy_from_others/land_sea_glacier_mask_phy"
    )
    lons, lats, data = get_accumulation_area()


    data = np.ma.masked_where(data < 1e2, data)
    data = np.ma.masked_where(land_sea_mask, data)

    x, y = bmap(lons, lats)
#    plot_default_topo(bmap, x, y, data)
    plot_usual_acc_area(bmap, x, y, data)
    pass



if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main()