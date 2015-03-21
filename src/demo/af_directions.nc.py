from netCDF4 import Dataset
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import application_properties

import matplotlib.pyplot as plt
from util import plot_utils

__author__ = 'huziy'

import numpy as np


def main():
    """

    """

    s_lons = [-3.38, 3.40, -12.45, -11.04, -2.92, 0.1, 32.55, 30.48, 23.6, 31.27, 15.31, 23.91, 17.51, 21.08]
    s_lats = [16.26, 11.88, 14.9, 13.91, 10.57, 6.2, 15.61, 19.18, -14.02, -21.13, -4.26, 4.97, -28.71, -28.69]

    assert len(s_lons) == len(s_lats)
    # path = "data/directions_Africa_Bessam_0.44/infocell_Africa.nc"
    path = "/home/huziy/skynet3_exec1/for_offline_routing/directions_africa_dx0.44deg_2.nc"
    ds = Dataset(path)

    dirs = ds.variables["flow_direction_value"][:]
    acc_area = ds.variables["accumulation_area"][:]

    lons2d = ds.variables["lon"][:]
    lats2d = ds.variables["lat"][:]

    lons2d[lons2d >= 180] -= 360

    min_lon = lons2d.min() + 6
    max_lon = lons2d.max() - 6
    min_lat = lats2d.min() + 11
    max_lat = lats2d.max() - 4

    #plot_utils.apply_plot_params(width_pt=None, width_cm=80)
    print max_lon

    fig = plt.figure(dpi=500, figsize=(11, 8.5))
    b = Basemap(projection="mill", llcrnrlon=min_lon,
                llcrnrlat=min_lat,
                urcrnrlon=max_lon, urcrnrlat=max_lat, resolution="i")

    x, y = b(lons2d, lats2d)
    b.drawcoastlines(linewidth=0.2)


    #    b.pcolormesh(x, y, np.ma.masked_where(dirs <= 0, dirs ))
    #    plt.colorbar()


    di_list = np.array([1, 1, 0, -1, -1, -1, 0, 1])
    dj_list = np.array([0, -1, -1, -1, 0, 1, 1, 1])

    delta_indices = np.log2(dirs[dirs > 0])
    delta_indices = delta_indices.astype(int)

    di = di_list[delta_indices]
    dj = dj_list[delta_indices]

    acc_area = np.ma.masked_where(acc_area < 0, acc_area)
    #img = b.pcolormesh(x, y, np.ma.log(acc_area))

    ax = plt.gca()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", "5%", pad="3%")

    #plt.colorbar(img, cax = cax)



    i_indices_1d = range(dirs.shape[0])
    j_indices_1d = range(dirs.shape[1])

    j_indices_2d, i_indices_2d = np.meshgrid(j_indices_1d, i_indices_1d)

    i_indices_2d_next = np.zeros_like(i_indices_2d)
    j_indices_2d_next = np.zeros_like(i_indices_2d)

    i_indices_2d_next[dirs > 0] = i_indices_2d[dirs > 0] + di
    j_indices_2d_next[dirs > 0] = j_indices_2d[dirs > 0] + dj

    for i, j, i_next, j_next in zip(i_indices_2d[dirs > 0], j_indices_2d[dirs > 0],
                                    i_indices_2d_next[dirs > 0], j_indices_2d_next[dirs > 0]):
        ax.add_line(Line2D([x[i, j], x[i_next, j_next]], [y[i, j], y[i_next, j_next]], linewidth=0.5))

    x1, y1 = b(s_lons, s_lats)
    b.scatter(x1, y1, c="r", linewidth=0, zorder=7, ax=ax)
    b.drawrivers(linewidth=0.5, color="#0cf5f8", zorder=8, ax=ax)
    # b.drawmeridians(np.arange(-10, 90,30))
    # b.drawparallels(np.arange(-50, 40, 5), labels=[1,1,1,1], linewidth=0.1)
    plt.tight_layout()
    # plt.show()

    b.readshapefile("/home/huziy/skynet3_exec1/other_shape/af_major_basins/af_basins", "basin",
                    linewidth=3, zorder=9, ax=ax
    )
    plt.savefig("with_station_riv_af_dirs_basin_1.0.pdf")

    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  