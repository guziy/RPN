from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedFormatter, ScalarFormatter

import os

from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import maskoceans
from domains.rotated_lat_lon import RotatedLatLon
import my_colormaps
from rpn import level_kinds
from rpn.rpn import RPN
from util import plot_utils

__author__ = 'huziy'

import numpy as np
import crcm5.analyse_hdf.common_plot_params as cpp
import matplotlib.pyplot as plt


def main():
    # plot_utils.apply_plot_params(20)
    folder = "/home/huziy/skynet3_rech1/from_guillimin"
    fname = "geophys_Quebec_0.1deg_260x260_with_dd_v6"
    path = os.path.join(folder, fname)

    rObj = RPN(path)

    mg = rObj.get_first_record_for_name_and_level("MG", level=0, level_kind=level_kinds.PRESSURE)
    # j2 = rObj.get_first_record_for_name("J2")


    levs = [0, 100, 200, 300, 500, 700, 1000, 1500, 2000, 2800]
    norm = BoundaryNorm(levs, len(levs) - 1)

    me = rObj.get_first_record_for_name_and_level("ME", level=0, level_kind=level_kinds.ARBITRARY)

    proj_params = rObj.get_proj_parameters_for_the_last_read_rec()

    print(me.shape)
    lons2d, lats2d = rObj.get_longitudes_and_latitudes_for_the_last_read_rec()

    lons2d[lons2d > 180] -= 360



    # me_to_plot = np.ma.masked_where(mg < 0.4, me)
    me_to_plot = me
    # print me_to_plot.min(), me_to_plot.max()


    rll = RotatedLatLon(**proj_params)
    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons2d, lats2d=lats2d)
    rObj.close()


    x, y = basemap(lons2d, lats2d)

    plt.figure()
    ax = plt.gca()
    # the_cmap = cm.get_cmap(name = "gist_earth", lut=len(levs) -1)
    # the_cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/topo_15lev.rgb")
    the_cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/OceanLakeLandSnow.rgb",
                                                        ncolors=len(levs) - 1)


    # new_cm = matplotlib.colors.LinearSegmentedColormap('colormap',new_dict,len(levs) - 1)

    me_to_plot = maskoceans(lons2d, lats2d, me_to_plot, resolution="l")
    basemap.contourf(x, y, me_to_plot,
                     cmap=the_cmap, levels=levs, norm=norm)




    # basemap.fillcontinents(color = "none", lake_color="aqua")
    basemap.drawmapboundary(fill_color='#479BF9')
    basemap.drawcoastlines(linewidth=0.5)
    basemap.drawmeridians(np.arange(-180, 180, 20), labels=[1, 0, 0, 1])
    basemap.drawparallels(np.arange(45, 75, 15), labels=[1, 0, 0, 1])


    basemap.readshapefile("data/shape/contour_bv_MRCC/Bassins_MRCC_latlon", name="basin", linewidth=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(ticks=levs, cax=cax)

    # basemap.scatter(x, y, color="k", s=1, linewidths=0, ax=ax, zorder=2)

    margin = 20
    x1 = x[margin, margin]
    x2 = x[-margin, margin]
    y1 = y[margin, margin]
    y2 = y[margin, -margin]
    pol_corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    ax.add_patch(Polygon(xy=pol_corners, fc="none", ls="dashed", lw=3))

    # plt.tight_layout()
    # plt.show()
    plt.savefig("free_domain_260x260.png", dpi=cpp.FIG_SAVE_DPI)



    pass


def plot_lake_fraction_field():
    folder = "/home/huziy/skynet3_rech1/geof_lake_infl_exp"
    fName = "geophys_Quebec_0.1deg_260x260_with_dd_v6"
    path = os.path.join(folder, fName)

    rObj = RPN(path)

    lkf = rObj.get_first_record_for_name_and_level(varname="VF", level=3, level_kind=level_kinds.ARBITRARY)

    proj_params = rObj.get_proj_parameters_for_the_last_read_rec()
    lons2d, lats2d = rObj.get_longitudes_and_latitudes_for_the_last_read_rec()

    lons2d[lons2d >= 180] -= 360
    rObj.close()

    rll = RotatedLatLon(**proj_params)

    margin = 20
    lons2d = lons2d[margin:-margin, margin:-margin]
    lats2d = lats2d[margin:-margin, margin:-margin]
    lkf = lkf[margin:-margin, margin:-margin]

    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons2d, lats2d=lats2d, resolution="l")
    x, y = basemap(lons2d, lats2d)

    fig = plt.figure()
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    df = 0.1
    levels = np.arange(0, 1.1, df)
    cMap = get_cmap("gist_ncar_r", len(levels) - 1)
    bn = BoundaryNorm(levels, cMap.N)

    basemap.drawmapboundary(fill_color="0.75")
    lkf_plot = maskoceans(lons2d, lats2d, lkf, inlands=False)
    print("Percentage of lakes in the sim domain: {}".format(lkf_plot.mean() * 100))

    img = basemap.pcolormesh(x, y, lkf_plot, norm=bn, cmap=cMap)
    basemap.drawcoastlines()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", "5%", pad="3%")
    cb = fig.colorbar(img, cax=cax, ticks=levels, orientation="horizontal")

    ax = fig.add_subplot(gs[0, 1])
    df1 = df
    levels1 = np.arange(0, 1.1, df1)
    cell_numms = np.zeros((len(levels1) - 1, ))

    left = levels[0]
    right = levels[1]

    lefts = []
    rights = []
    lkf_land = lkf[lkf > 0.01]
    for i in range(len(cell_numms)):
        cell_numms[i] = ((lkf_land > left) & (lkf_land <= right)).astype(int).sum()
        lefts.append(left)
        rights.append(right)
        left += df1
        right += df1

    assert isinstance(ax, Axes)
    ax.bar(lefts, cell_numms, width=df1)

    # ax.semilogy(rights, cell_numms)
    ax.xaxis.set_ticks(levels)
    ax.yaxis.set_ticks(np.arange(1000, 10000, 1000))
    sf = ScalarFormatter(useMathText=True)
    sf.set_powerlimits([-2, 1])
    ax.yaxis.set_major_formatter(sf)

    ax.grid("on")
    ax.set_xlabel("fraction")
    ax.set_ylabel("# gridcells")

    plt.show()
    fig.tight_layout()
    fig.savefig("lake_fractions_220x220_0.1deg.jpeg")
    plt.show()

    pass


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    # plot_utils.apply_plot_params(width_pt=None, width_cm=45, height_cm=30, font_size=20)
    # plot_lake_fraction_field()
    # plot_utils.apply_plot_params(font_size=10, width_pt=None, width_cm=20.0 / 7.0, height_cm=10.0 / 7.0)

    plot_utils.apply_plot_params(font_size=18, width_pt=None, width_cm=20.0 / 1.2, height_cm=17.0 / 1.2)
    main()
    print("Hello world")