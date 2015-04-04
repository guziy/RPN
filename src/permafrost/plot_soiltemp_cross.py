import os
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import date2num, num2date
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from mpl_toolkits.basemap import Basemap
from scipy.spatial.kdtree import KDTree
from . import draw_regions
from .active_layer_thickness import CRCMDataManager
from util.geo import lat_lon
from matplotlib import cm

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt

def get_flat_index(lon, lat, the_kd_tree ):
    """
    :type the_kd_tree: KDTree
    """
    x0,y0,z0 = lat_lon.lon_lat_to_cartesian(lon, lat)
    d, i = the_kd_tree.query((x0,y0,z0))
    return i


def main_for_spinup():
        #dm = CRCMDataManager(data_folder="/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_ERA40-Int_old_snow_cond")
    dm = CRCMDataManager(data_folder="/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NA_1.0deg_soil_spinup2")
    start_year = 2056
    end_year = 2156
    T0 = 273.15

    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NA_1.0deg_soil_spinup2" #for coordinates
    coord_file = os.path.join(sim_data_folder, "pmNA_1.0deg_soil_spinup2_227810_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74

    )


#    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1" #for coordinates
#    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
#    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
#        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74
#    )
    assert isinstance(basemap, Basemap)

    #basemap.transform_scalar()

    #basemap = Basemap()
    lons2d[lons2d > 180] -= 360

    x, y, z = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
    the_kd_tree = KDTree(list(zip(x, y, z)))
    lon_p = -107-13.0/60.0 #-96.65
    lat_p = 26.0 + 10.0/60.0#29.75




    levels = dm.level_heights

    years = range(start_year, end_year + 1)

    all_months, t4d = dm.get_monthly_mean_soiltemps(year_range=years)
    t4d -= T0
    levs2d, all_months_2d = np.meshgrid(levels, date2num(all_months))

    dt4d = np.zeros(t4d.shape)
    dt4d[1:,:,:,:] = np.abs( t4d[1:,:,:,:] - t4d[:-1,:,:,:] )




    ids = range(10)
    lons_p = [-124.306,-116.205,-112.291,-110.215,-87.881,-87.7899,-87.6967,-86.2741,-86.1661,-85.5195]
    lats_p = [60.2532,61.9556,61.483,60.7783,54.8701,55.307,55.7438,55.1947,55.6304,55.1322]

    for the_i, lon_p, lat_p in zip(ids, lons_p, lats_p):
        flat_idx = get_flat_index(lon_p, lat_p, the_kd_tree = the_kd_tree)

        ind = np.where( (lons2d == lons2d.flatten()[flat_idx]) & (lats2d == lats2d.flatten()[flat_idx]))

        ix = ind[0][0]
        jy = ind[1][0]

        t1 = t4d[:,ix, jy,:]
        dt1 = dt4d[:,ix, jy, :]


        plt.figure()
        clevs = [0, 0.001, 0.01,0.02,0.03,0.1,0.2,0.3,0.4,0.8,1,2,3,4,5,6,7,8, 10, 15]
        norm = BoundaryNorm(boundaries=clevs, ncolors=len(clevs) - 1)
        cmap = cm.get_cmap("jet", len(clevs) - 1)
        plt.contourf(all_months_2d, levs2d, dt1, levels = clevs, cmap = cmap, norm = norm)
        plt.colorbar(ticks = clevs, norm = norm)
        #plt.xticks(years)
        #plt.yticks(levels)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: num2date(float(x)).strftime("%Y")
        ))
        ax.set_title("Differences, {0}, {1}-{2}".format(the_i, start_year, end_year))
        plt.savefig("cross_spinup_diffs_{0}.png".format(the_i))

        plt.figure()
        clevs = [-25,-20,-10,-5,-1,0,1,5,10,20,25]
        norm = BoundaryNorm(boundaries=clevs, ncolors=len(clevs) - 1)
        cmap = cm.get_cmap("jet", len(clevs) - 1)
        plt.contourf(all_months_2d, levs2d, t1, levels = clevs, cmap = cmap, norm = norm)
        plt.colorbar(ticks = clevs, norm = norm)
        #plt.xticks(years)
        #plt.yticks(levels)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, pos: num2date(float(x)).strftime("%Y")
        ))
        ax.set_title("Values, {0}, {1}-{2}".format(the_i, start_year, end_year))
        plt.savefig("cross_spinup_vals_{0}.png".format(the_i))


    #show point pos
    plt.figure()
    x, y = basemap(lon_p, lat_p)
    basemap.scatter(x,y, c = "r", s = 200, zorder = 5)
    basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
            linewidth=1.5)

    basemap.drawcoastlines()
    plt.savefig("position.png")


def main():
    #dm = CRCMDataManager(data_folder="/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_ERA40-Int_old_snow_cond")
    dm = CRCMDataManager(data_folder="/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1")
    start_year = 1981
    end_year = 2005
    T0 = 273.15

    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1" #for coordinates
    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74
    )


#    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1" #for coordinates
#    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
#    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
#        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74
#    )
    assert isinstance(basemap, Basemap)

    #basemap.transform_scalar()

    #basemap = Basemap()
    lons2d[lons2d > 180] -= 360

    x, y, z = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
    the_kd_tree = KDTree(list(zip(x, y, z)))
    lon_p = -130.97578#-132.3739 #-113.25241, #-100.0
    lat_p = 66.152588#66.360443 #59.584538 #60.0
    #flat_idx = get_flat_index(lon_p, lat_p, the_kd_tree = the_kd_tree)
    i = 44
    j = 125


    levels = dm.level_heights[1:10]

    years = range(start_year, end_year + 1)
    t1 = np.zeros((len(years), len(levels)))
    for iy, y in enumerate(years):
        t3d = dm.get_annual_mean_3d_field(var_name="I0", year=y)
        for ilev, lev in enumerate(levels):
            t1[iy, ilev] = t3d[i,j,ilev]#.flatten()[flat_idx]



    #t1 = np.mean( np.mean(t3ds, axis = 1), axis = 1 )

    levs2d, y2d = np.meshgrid(levels, years)

    plt.figure()
    clevs = [-5,-4, -3,-1,-0.5,-0.2,-0.1,0, 0.1,0.2,0.3,0.4,0.8,1,2,3,4,5,6]
    norm = BoundaryNorm(boundaries=clevs, ncolors=len(clevs) - 1)
    cmap = cm.get_cmap("jet", len(clevs) - 1)
    plt.contourf(y2d, levs2d, t1 - T0, levels = clevs, cmap = cmap, norm = norm)
    #plt.pcolormesh(y2d, levs2d, t1 - T0,  cmap = cmap, norm = norm)
    plt.colorbar(ticks = clevs, norm = norm)
    #plt.xticks(years)
    #plt.yticks(levels)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.savefig("cross_check.png")


    #show point pos
    plt.figure()
    x, y = basemap(lon_p, lat_p)
    basemap.scatter(x,y, c = "r", s = 200, zorder = 5)
    basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
            linewidth=1.5)

    basemap.drawcoastlines()
    plt.savefig("position.png")


    plt.show()


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()


    #main()
    main_for_spinup()
    print("Hello world")
  