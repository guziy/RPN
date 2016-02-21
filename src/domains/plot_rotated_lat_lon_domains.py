import os
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.basemap import Basemap
from data.grid_params import GridParams
from domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
from util import plot_utils

__author__ = 'huziy'

import numpy as np

import matplotlib.pyplot as plt


def plot_north_cross(lon0, lat0, basemap, ax=None):
    """
    :type ax: Axes
    """
    d = 2
    lonw = lon0 - d
    lone = lon0 + d
    latn = lat0 + d
    lats = lat0 - d

    west_point = basemap(lonw, lat0)
    east_point = basemap(lone, lat0)
    north_point = basemap(lon0, latn)
    south_point = basemap(lon0, lats)

    # hor_line = Line2D([xe, xw], [ye, yw], color="k")
    ax.add_line(FancyArrowPatch(south_point, north_point, arrowstyle="->", mutation_scale=30, linewidth=4))
    ax.add_line(FancyArrowPatch(east_point, west_point, arrowstyle="-", mutation_scale=30, linewidth=4))
    ax.annotate("N", xy=north_point, va="bottom", ha="center",
                font_properties=FontProperties(weight="bold", size=20))

    pass


NA_RIVERS_SHAPE_PATH = "/home/huziy/skynet3_exec1/other_shape/10m-rivers-lake-centerlines/ne_10m_rivers_lake_centerlines"


def get_lons_lats_using_grid_params(g_params, rot_latlon):
    """
    :type g_params: GridParams
    :type rot_latlon: RotatedLatLon

    Note: when specifying iref and jref in the grid paraneters be carefull, since
    in Python I allways consider that indices start from 0
    """

    lons2d = np.zeros((g_params.nx, g_params.ny))
    lats2d = np.zeros((g_params.nx, g_params.ny))

    iref = g_params.iref
    jref = g_params.jref

    lonr = g_params.lonr
    latr = g_params.latr

    dx = g_params.dx
    dy = g_params.dy

    for i in range(g_params.nx):
        for j in range(g_params.ny):
            loni = lonr + (i - iref) * dx
            latj = latr + (j - jref) * dy

            lons2d[i, j], lats2d[i, j] = rot_latlon.toGeographicLonLat(loni, latj)

    return lons2d, lats2d
    pass


def plot_domain_for_different_margins(path, margins=None):
    if not margins: margins = [20, 40, 60]
    rpnObj = RPN(path)

    lons2d, lats2d = rpnObj.get_longitudes_and_latitudes()

    # projection parameters
    lon_1 = -68
    lat_1 = 52
    lon_2 = 16.65
    lat_2 = 0.0

    rot_lat_lon = RotatedLatLon(lon1=lon_1, lat1=lat_1, lon2=lon_2, lat2=lat_2)
    xll, yll = rot_lat_lon.toProjectionXY(lons2d[0, 0], lats2d[0, 0])
    xur, yur = rot_lat_lon.toProjectionXY(lons2d[-1, -1], lats2d[-1, -1])

    if xll < 0: xll += 360.0
    if xur < 0: xur += 360.0

    nx, ny = lons2d.shape

    dx = (xur - xll) / float(nx - 1)
    dy = (yur - yll) / float(ny - 1)

    print(dx, dy)
    print(xur, yur, xll, yll)

    x1 = xll - dx / 2.0
    y1 = yll - dy / 2.0
    x2 = xur + dx / 2.0
    y2 = yur + dy / 2.0

    x1lon, y1lat = rot_lat_lon.toGeographicLonLat(x1, y1)
    x2lon, y2lat = rot_lat_lon.toGeographicLonLat(x2, y2)

    llcrnrlon, llcrnrlat = rot_lat_lon.toGeographicLonLat(x1 - dx, y1 - dx)
    urcrnrlon, urcrnrlat = rot_lat_lon.toGeographicLonLat(x2 + dx, y2 + dx)

    basemap = Basemap(projection="omerc",
                      lon_1=lon_1, lat_1=lat_1,
                      lon_2=lon_2, lat_2=lat_2,
                      llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                      urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat, no_rot=True, resolution="l")

    basemap.drawcoastlines()
    basemap.drawrivers()

    x1, y1 = basemap(x1lon, y1lat)
    x2, y2 = basemap(x2lon, y2lat)

    # add rectangle for the grid 220x220
    #    r1 = Rectangle((x1, y1), x2-x1, y2-y1, facecolor="none", edgecolor="r",  linewidth=5  )

    ax = plt.gca()
    assert isinstance(ax, Axes)
    #    xr1_label, yr1_label = rot_lat_lon.toGeographicLonLat(xur - 2 * dx, yll + 2 * dy)
    #    xr1_label, yr1_label = basemap( xr1_label, yr1_label )
    #    ax.annotate("{0}x{1}".format(nx, ny), xy = (xr1_label, yr1_label), va = "bottom", ha = "right", color = "r")
    #    assert isinstance(ax, Axes)
    #    ax.add_patch(r1)

    margins_all = [0] + margins

    for margin in margins_all:
        # mfree = margin - 20
        xlli = xll + margin * dx
        ylli = yll + margin * dy
        xuri = xur - margin * dx
        yuri = yur - margin * dy

        x1lon, y1lat = rot_lat_lon.toGeographicLonLat(xlli, ylli)
        x2lon, y2lat = rot_lat_lon.toGeographicLonLat(xuri, yuri)

        x1, y1 = basemap(x1lon, y1lat)
        x2, y2 = basemap(x2lon, y2lat)

        ri = Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor="none", edgecolor="r", linewidth=5)
        ax.add_patch(ri)

        xri_label, yri_label = rot_lat_lon.toGeographicLonLat(xlli + 2 * dx, yuri - 2 * dy)
        xri_label, yri_label = basemap(xri_label, yri_label)
        ax.annotate("{0}x{1}\nmarg. = {2}".format(nx - margin * 2, ny - margin * 2, margin + 20),
                    xy=(xri_label, yri_label),
                    va="top", ha="left", color="k", backgroundcolor="w")

    plt.show()


def plot_domain_using_coords_from_file(path=""):
    fig = plt.figure()
    assert isinstance(fig, Figure)
    rpnObj = RPN(path)

    lons2d, lats2d = rpnObj.get_longitudes_and_latitudes()

    basemap = Basemap(projection="omerc", lon_1=-68, lat_1=52,
                      lon_2=16.65, lat_2=0.0, llcrnrlon=lons2d[0, 0], llcrnrlat=lats2d[0, 0],
                      urcrnrlon=lons2d[-1, -1], urcrnrlat=lats2d[-1, -1], no_rot=True)

    # basemap.drawcoastlines()



    rot_lat_lon_proj = RotatedLatLon(lon1=-68, lat1=52, lon2=16.65, lat2=0.0)

    g_params = GridParams(lonr=180, latr=0, iref=45, jref=41, dx=0.5, dy=0.5, nx=86, ny=86)

    lons2d_1, lats2d_1 = get_lons_lats_using_grid_params(g_params, rot_lat_lon_proj)

    basemap = Basemap(projection="omerc", lon_1=-68, lat_1=52,
                      lon_2=16.65, lat_2=0.0, llcrnrlon=lons2d_1[18, 18], llcrnrlat=lats2d_1[18, 18],
                      urcrnrlon=lons2d_1[-1, -1], urcrnrlat=lats2d_1[-1, -1], no_rot=True, resolution="i")

    basemap.drawcoastlines(linewidth=0.4)
    basemap.drawrivers()
    # basemap.drawmeridians(np.arange(-180, 0, 20))


    x, y = basemap(lons2d, lats2d)
    basemap.scatter(x, y, c="r", linewidths=0, s=1.0)
    print(x.shape)

    xll_big, yll_big = g_params.get_ll_point(marginx=20, marginy=20)
    xll_big -= g_params.dx / 2.0
    yll_big -= g_params.dy / 2.0
    xll_big, yll_big = rot_lat_lon_proj.toGeographicLonLat(xll_big, yll_big)
    xll_big, yll_big = basemap(xll_big, yll_big)

    xur_big, yur_big = g_params.get_ur_point(marginx=20, marginy=20)
    xur_big += g_params.dx / 2.0
    yur_big += g_params.dy / 2.0
    xur_big, yur_big = rot_lat_lon_proj.toGeographicLonLat(xur_big, yur_big)
    xur_big, yur_big = basemap(xur_big, yur_big)

    margin = 20

    # plot 0.25 degree grid
    g_params = GridParams(lonr=180, latr=0, iref=71, jref=63, dx=0.25, dy=0.25, nx=133, ny=133)
    lons2d_2, lats2d_2 = get_lons_lats_using_grid_params(g_params, rot_lat_lon_proj)
    x2, y2 = basemap(lons2d_2[margin:-margin, margin:-margin], lats2d_2[margin:-margin, margin:-margin])
    basemap.scatter(x2, y2, c="g", linewidth=0, marker="s", s=7.5)

    # plot 0.5 degree grid using the output file
    # debug
    rObj1 = RPN("/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes/pm1985010100_00000000p")
    lons2d_1, lats2d_1 = rObj1.get_longitudes_and_latitudes()

    # x1, y1 = basemap(lons2d_1[margin:-margin,margin:-margin], lats2d_1[margin:-margin,margin:-margin])
    x1, y1 = basemap(lons2d_1, lats2d_1)

    print(x1.shape, lons2d_1[0, 0], lats2d_1[0, 0])

    basemap.scatter(x1, y1, c="b", linewidths=0, s=10)

    dx1 = (x1[1, 0] - x1[0, 0]) / 2.0
    dy1 = (y1[0, 1] - y1[0, 0]) / 2.0

    rbig = Rectangle((xll_big, yll_big), xur_big - xll_big,
                     yur_big - yll_big, linewidth=2, edgecolor="b",
                     facecolor="none"
                     )

    ax = plt.gca()
    assert isinstance(ax, Axes)
    # ax.add_patch(rsmall)
    ax.add_patch(rbig)

    # draw north arrow
    plot_north_cross(-45, 45, basemap, ax=ax)

    # zoom to a region
    axins = zoomed_inset_axes(ax, 4, loc=1)  # zoom = 6
    basemap.drawcoastlines(ax=axins)
    basemap.drawrivers(ax=axins)
    basemap.scatter(x, y, c="r", linewidths=0, s=5, ax=axins)
    basemap.scatter(x2, y2, c="g", marker="s", linewidth=0, s=15, ax=axins)
    basemap.scatter(x1, y1, c="b", linewidths=0, s=25, ax=axins)

    # subregion to zoom in
    nx, ny = lons2d.shape
    part = 3
    xins_ll = lons2d[nx / part, ny / part]
    yins_ll = lats2d[nx / part, ny / part]
    xins_ur = lons2d[nx / part + 40, ny / part + 40]
    yins_ur = lats2d[nx / part + 40, ny / part + 40]

    xins_ll, yins_ll = basemap(xins_ll, yins_ll)
    xins_ur, yins_ur = basemap(xins_ur, yins_ur)

    axins.set_xlim(xins_ll, xins_ur)
    axins.set_ylim(yins_ll, yins_ur)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", linewidth=2)

    fig.tight_layout(pad=0.8)
    fig.savefig("high_low_res_domains.png")

    pass


def main():
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_198501_198612_0.1deg"
    coord_file = os.path.join(data_path, "pm1985010100_00000000p")

    plot_domain_using_coords_from_file(path=coord_file)

    # plot_domain_for_different_margins(coord_file)

    pass


if __name__ == "__main__":
    import application_properties

    plot_utils.apply_plot_params(width_pt=None, width_cm=40, height_cm=40, font_size=15)
    application_properties.set_current_directory()
    main()
    print("Hello world")
