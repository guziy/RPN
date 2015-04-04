from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

__author__ = 'huziy'

import numpy as np
from . import cehq_station
from permafrost import draw_regions
import matplotlib.pyplot as plt
import application_properties

def main():
    stations = cehq_station.read_station_data(folder="data/cehq_levels")

    lons_s = []
    lats_s = []
    values = []
    for s in stations:
        lon, lat = s.longitude, s.latitude
        lons_s.append(lon)
        lats_s.append(lat)
        values.append(s.get_mean_value())

    lons_s = np.array(lons_s)
    lats_s = np.array(lats_s)

    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(
        file_path="data/from_guillimin/vary_lake_level1/pm1985010100_00000000p",
        lon1=-68, lat1=52, lon2=16.65, lat2=0, resolution="h"
    )

    x, y = b(lons_s, lats_s)


    fig = plt.figure()
    ax = plt.gca()
    b.warpimage()
    b.scatter(x, y, c = "r", zorder = 2)
    b.drawcoastlines(linewidth = 0.2)
    #draw inset axes
    axins = zoomed_inset_axes(plt.gca(), 2, loc=1)
    b.scatter(x, y, c = "r", zorder = 2, ax = axins)
    b.warpimage(ax = axins)

    ins_ll_lon = -76
    ins_ll_lat = 45

    ins_ur_lon = -68
    ins_ur_lat = 48

    x_ll, y_ll = b(ins_ll_lon, ins_ll_lat)
    x_ur, y_ur = b(ins_ur_lon, ins_ur_lat)
    axins.set_xlim(x_ll, x_ur)
    axins.set_ylim(y_ll, y_ur)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    fig.savefig("cehq_level_stations.png")

    fig = plt.figure()
    ax = plt.gca()
    b.warpimage()
    b.scatter(x, y, c = values, zorder = 2)
    plt.colorbar()
    b.drawcoastlines(linewidth = 0.2)

    #draw inset axes
    axins = zoomed_inset_axes(plt.gca(), 4, loc=1)
    b.scatter(x, y, c = values, zorder = 2, ax = axins)
    b.warpimage(ax = axins)
    b.drawcoastlines(linewidth = 0.2, ax = axins)

    ins_ll_lon = -73
    ins_ll_lat = 45

    ins_ur_lon = -70
    ins_ur_lat = 47

    x_ll, y_ll = b(ins_ll_lon, ins_ll_lat)
    x_ur, y_ur = b(ins_ur_lon, ins_ur_lat)
    axins.set_xlim(x_ll, x_ur)
    axins.set_ylim(y_ll, y_ur)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    fig.savefig("cehq_levels.png")



if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print("Hello world")
  