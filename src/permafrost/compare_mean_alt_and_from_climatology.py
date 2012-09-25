from netCDF4 import Dataset
from descartes.patch import PolygonPatch
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import date2num, num2date
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from scipy.spatial.kdtree import KDTree
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
import draw_regions
import my_colormaps
from cross_plotter import SoundingPlotter
from rpn.rpn import RPN
from util import plot_utils
import matplotlib as mpl
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
import os
from active_layer_thickness import CRCMDataManager
import matplotlib.pyplot as plt


from matplotlib.projections.geo import Transform
from osgeo import ogr, osr
import descartes
from shapely import wkb

def do_stats_plots(x,y,basemap, hc, hm, hm2d, permafrost_mask):
    the_mask = hc.mask | hm2d.mask
    the_min = np.ma.masked_all(the_mask.shape)
    the_max = np.ma.masked_all(the_mask.shape)
    the_std = np.ma.masked_all(the_mask.shape)

    all_axes = []
    all_img = []
    gs = gridspec.GridSpec(3,1)
    fig = plt.figure()
    assert isinstance(fig, Figure)
    ax = fig.add_subplot(gs[0,0])
    the_min[~the_mask] = np.ma.min(hm[:,~the_mask], axis=0)
    print the_min.shape
    img = basemap.contourf(x,y, the_min, ax = ax)
    all_img.append(img)
    all_axes.append(ax)
    ax.set_title("Min")

    ax = fig.add_subplot(gs[1,0])
    the_max[~the_mask] = np.ma.max(hm[:,~the_mask], axis=0)
    img = basemap.contourf(x,y, the_max, ax = ax)
    all_img.append(img)
    all_axes.append(ax)
    ax.set_title("Max")

    ax = fig.add_subplot(gs[2,0])
    the_std[~the_mask] = np.ma.std(hm[:,~the_mask], axis=0)
    print the_std.shape

    img = basemap.contourf(x,y, the_std, ax = ax)
    all_img.append(img)
    all_axes.append(ax)
    ax.set_title("Std")

    for the_ax, the_img in zip( all_axes, all_img ):
        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=0.5)
        divider = make_axes_locatable(the_ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = fig.colorbar(the_img,  cax = cax)
        CS = basemap.contour(x,y, permafrost_mask, levels = [0,1,2,3,4],
            ax = the_ax, colors = "k", linewidth= 5)


    fig.tight_layout()
    fig.savefig("alt_b1_stats.png")

    pass

def plot_mean_alt_from_jpp_results():
    start_year = 1981
    end_year = 2008
    path = "data/alts_by_jpp/MonTS_NA_ERA40_ALT_{0}_{1}".format(start_year, end_year)

    rObj = RPN(path)

    altt = rObj.get_all_time_records_for_name(varname="FALT")
    alt = np.mean( np.array(altt.values()), axis = 0)
    rObj.close()



    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"
    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74
    )
    assert isinstance(basemap, Basemap)

    lons2d[lons2d > 180] -= 360

    x, y = basemap(lons2d, lats2d)


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3)
    alt = np.ma.masked_where(mask_cond, alt)

    fig = plt.figure()
    assert isinstance(fig, Figure)


    h_max = 10
    cmap = my_colormaps.get_lighter_jet_cmap(ncolors=10) #cm.get_cmap("jet",10)
    bounds = [0,0.1,0.5,1,2,3,5,8,9,10,11]
    norm = BoundaryNorm(boundaries=bounds,ncolors=len(bounds), clip=True)
    #norm = None

    cmap.set_over(cmap(1.0))
    clevels = np.arange(0,h_max+1,1)
    gs = gridspec.GridSpec(3,1)

    ax = fig.add_subplot(gs[0,0])
    assert isinstance(ax, Axes)
    hc = np.ma.masked_where(mask_cond | (np.min(altt.values(), axis = 0) < 0), alt)
    #hc = np.ma.masked_where( (hc < 0), hc)
    img = basemap.pcolormesh(x, y, hc, cmap = cmap, vmax = h_max, norm=norm)
    ax.set_title("ALT, JPP ({0} - {1}) \n".format(start_year, end_year))

    basemap.drawcoastlines(ax = ax, linewidth=0.5)
    basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
            ax=ax, linewidth=1.5)


    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img,  cax = cax, extend = "max", ticks = bounds)
    cax.set_title("m \n")


    fig.tight_layout(w_pad=0.0)


    fig.savefig("alt_jpp_current.png")




def get_zone_polygons(path = "", basemap = None):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(path, 0)
    layer = dataStore.GetLayer(0)
    latlong = osr.SpatialReference()
    latlong.ImportFromProj4("+proj=latlong")
    result = []

    feature = layer.GetNextFeature()
    while feature:
        geom = feature.GetGeometryRef()
        geom.TransformTo(latlong)


        polygon = wkb.loads(geom.ExportToWkb())

        print polygon.geom_type
        print polygon.boundary.geom_type

        bound = polygon.boundary

        if isinstance(polygon, MultiPolygon):
            feature = layer.GetNextFeature()
            continue
            the_lines = bound
        else:
            the_lines = [bound]

        x = []
        y = []


        for line in the_lines:
            x += map(lambda w: w[0], line.coords )
            y += map(lambda w: w[1], line.coords )

        x1,y1 = basemap(x,y)
        polygon = Polygon(zip(x1,y1))
        p = PolygonPatch(polygon, edgecolor = "k")
        #p.set_transform(to_map_proj)

        result.append(p)
        feature = layer.GetNextFeature()


    dataStore.Destroy()
    return result



def main():
    start_year = 1981
    end_year = 2008

    #mean alt
    path_to_yearly = "alt_era_b1_yearly.nc"
    ds = Dataset(path_to_yearly)

    hm = ds.variables["alt"][:]
    years = ds.variables["year"][:]
    years_sel = np.where(( start_year <= years ) & (years <= end_year))[0]
    print years_sel

    hm = hm[np.array(years_sel),:,:]
    print hm.shape

    good_points = ~np.any(hm < 0, axis = 0)

    hm2d = np.ma.masked_all(good_points.shape)


    hm2d[good_points] = np.mean( hm[ : , good_points],
                        axis = 0)


    #alt from climatology
    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"
#    dm = CRCMDataManager(data_folder=sim_data_folder)
#    hc = dm.get_alt_using_monthly_mean_climatology(xrange(start_year,end_year+1))



    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74
    )
    assert isinstance(basemap, Basemap)

    #basemap.transform_scalar()

    #basemap = Basemap()
    lons2d[lons2d > 180] -= 360

    x, y = basemap(lons2d, lats2d)
    #x = (x[1:,1:] + x[:-1, :-1]) /2.0


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3)

    #plot_utils.apply_plot_params(width_pt=None, width_cm=25,height_cm=35, font_size=12)
    fig = plt.figure()
    assert isinstance(fig, Figure)


    h_max = 10
    cmap = my_colormaps.get_lighter_jet_cmap(ncolors=10) #cm.get_cmap("jet",10)
    bounds = [0,0.1,0.5,1,2,3,5,8,9,10,11]
    norm = BoundaryNorm(boundaries=bounds,ncolors=len(bounds), clip=True)

    cmap.set_over(cmap(1.0))
    clevels = np.arange(0,h_max+1,1)
    gs = gridspec.GridSpec(1,1)

    all_axes = []
    all_img = []

    print basemap(-96.999, 68.42)
    print basemap(-2.4e7,6.3e6, inverse = True)
    ax = fig.add_subplot(gs[0,0])
    hm2d = np.ma.masked_where(mask_cond, hm2d)
    img = basemap.pcolormesh(x, y, hm2d, cmap = cmap, vmax = h_max, norm=norm)
    #img = basemap.contourf(x, y, hm2d, levels = clevels, cmap = cmap)
    ax.set_title("Mean ALT")
    all_axes.append(ax)
    all_img.append(img)
    print("hm2d(min,max) = ",hm2d.min(), hm2d.max())


#    ax = fig.add_subplot(gs[1,0])
#    hc = np.ma.masked_where(hc < 0, hc)
#    hc = np.ma.masked_where(mask_cond | (hc > h_max) | hm2d.mask, hc)
#    img = basemap.contourf(x, y, hc, levels = clevels)
#    all_img.append(img)
#    all_axes.append(ax)
#    ax.set_title("ALT from climatology")
#    print("hc(min,max) = ",hc.min(), hc.max())


#    ax = fig.add_subplot(gs[2,0])
#    delta = hm2d - hc
#    delta = np.ma.masked_where(hc.mask | hm2d.mask, delta)
#    img = basemap.contourf(x, y, delta, levels = np.arange(-1,1.2,0.2),ax = ax,
#        cmap = my_colormaps.get_red_blue_colormap())
#    all_img.append(img)
#    all_axes.append(ax)
#    ax.set_title("Mean - Derived from climatology")


    #print(np.where((hm2d < hc) & ~(hc.mask | hm2d.mask)))



    #zones = get_zone_polygons(path = "data/permafrost/permaice.shp", basemap=basemap)
    permafrost_mask = np.ma.masked_where((permafrost_mask < 0)|(permafrost_mask >= 4), permafrost_mask)
    for the_ax, the_img in zip( all_axes, all_img ):
        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=0.5)
        divider = make_axes_locatable(the_ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = fig.colorbar(the_img,  cax = cax, extend = "max", ticks = bounds)
        #CS = basemap.contour(x,y, permafrost_mask, levels = [1,2],
        #     ax = the_ax, colors = "k", linewidth= 5)
        #the_ax.clabel(CS,colors = 'k', fmt="%d" , fontsize=8)
        basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
            ax=the_ax, linewidth=1.5)


        #for p in zones:
        #    the_ax.add_patch(p)

    fig.tight_layout()
    #cax_to_hide.set_visible(False)
    fig.savefig("alt_b1.png")
    plt.show()
    do_stats = False
    if not do_stats:
        return
    #do_stats_plots(x,y,basemap, hc, hm, hm2d, permafrost_mask)


def get_alt_using_nyear_rule(hct, nyears):
    """
    hct(year, nlon,nlat)
    """
    nt, nx, ny = hct.shape
    result = -np.ones((nx, ny))
    result[:,:] = np.inf

    stdev = np.std(hct, axis = 0)
    print stdev.min(), stdev.max()

    for t in xrange(nt - nyears + 1):
        hmax = np.max(hct[t:t+nyears,:,:], axis=0)
        cond = (result >= hmax) & (hmax >= 0)
        if np.any(cond):
            result[cond] = hmax[cond]
    result[result == np.inf] = -1
    return result



    pass


def plot_current_alts_nyear_rule(nyear = 2):
    start_year = 1981
    end_year = 2008

    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"

    sim_names = ["ERA40", "MPI","CanESM"]
    all_data_f = "/home/huziy/skynet1_rech3/cordex/for_Samira"
    simname_to_path = {
        "ERA40": os.path.join(all_data_f, "alt_era_b1_yearly.nc"),
        "MPI": os.path.join(all_data_f, "alt_mpi_b1_yearly.nc"),
        "CanESM": os.path.join(all_data_f, "alt_canesm_b1_yearly.nc")

    }

    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74
    )
    assert isinstance(basemap, Basemap)

    #basemap.transform_scalar()

    #basemap = Basemap()
    lons2d[lons2d > 180] -= 360

    x, y = basemap(lons2d, lats2d)
    #x = (x[1:,1:] + x[:-1, :-1]) /2.0


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3)

#    plot_utils.apply_plot_params(width_pt=None, width_cm=20, height_cm=40, font_size=25)
    fig = plt.figure()
    assert isinstance(fig, Figure)


    h_max = 10
    cmap = my_colormaps.get_lighter_jet_cmap(ncolors=10) #cm.get_cmap("jet",10)
    bounds = [0,0.1,0.5,1,2,3,5,8,9,10,11]
    norm = BoundaryNorm(boundaries=bounds,ncolors=len(bounds), clip=True)
    cmap.set_over(cmap(1.0))
    clevels = np.arange(0,h_max+1,1)
    gs = gridspec.GridSpec(3,1)


    all_axes = []
    all_img = []


    i = 0
    hc_list = []
    hct_list = []

    for name in sim_names:
        path = simname_to_path[name]


        #select data and needed alt
        ds = Dataset(path)
        years = ds.variables["year"][:]
        hct = ds.variables["alt"][(years >= start_year) & (years <= end_year),:,:]
        hct_list.append(hct)
        print "hct.shape = ", hct.shape
        #hc = get_alt_using_nyear_rule(hct, nyears = nyear)
        hc = np.mean(hct, axis = 0)


        hc_list.append(hc)
        ax = fig.add_subplot(gs[i,0])
        assert isinstance(ax, Axes)
        hc = np.ma.masked_where(mask_cond | (np.min(hct, axis = 0) < 0), hc)
        #hc = np.ma.masked_where( (hc < 0), hc)
        img = basemap.pcolormesh(x, y, hc, cmap = cmap, vmax = h_max, norm=norm)
        if not i:
            ax.set_title("ALT, mean ({0} - {1}) \n".format(start_year, end_year))
        i += 1
        ax.set_ylabel(name)
        all_axes.append(ax)
        all_img.append(img)



    i = 0
    axs_to_hide = []
    #zones and coastlines
    for the_ax, the_img in zip(all_axes, all_img):

        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=0.5)
        basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
                ax=the_ax, linewidth=1.5)


        divider = make_axes_locatable(the_ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = fig.colorbar(the_img,  cax = cax, extend = "max", ticks = bounds)
        cax.set_title("m \n")


        if i != 2:
            axs_to_hide.append(cax)
        i += 1

    fig.tight_layout(w_pad=0.0)

    for the_ax in axs_to_hide:
        the_ax.set_visible(False)

    fig.savefig("alt_mean_current.png")

    #print ALT for selected points
    site_names = ["S","K","T"]
    sel_lons = [-75.646, -65.92, -69.95]
    sel_lats = [62.197, 58.709, 58.67]

    xo,yo,zo = lat_lon.lon_lat_to_cartesian(sel_lons, sel_lats)

    xi, yi, zi = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
    ktree = KDTree(zip(xi,yi,zi))
    dists, indexes =  ktree.query(zip(xo,yo,zo))

    for name, data, the_hct in zip(sim_names, hc_list, hct_list):
        print name
        flat_data = data.flatten()

        for p_name, ind in zip(site_names, indexes):
            in_data = []
            for t in xrange(the_hct.shape[0]):
                in_data.append(the_hct[t,:,:].flatten()[ind])

            print ",".join(map( lambda x: "{0:.1f}".format(float(x)), in_data))
            print p_name, "{0:.1f} m".format(float(flat_data[ind]))
        print "--" * 10




def plot_future_alts():
    start_year = 2041
    end_year = 2070

    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"

    sim_names = ["MPI","CanESM"]
    simname_to_path = {
        "MPI": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_MPI_B1",
        "CanESM": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1"

    }

    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74,
        anchor="W"
    )
    assert isinstance(basemap, Basemap)

    #basemap.transform_scalar()

    #basemap = Basemap()
    lons2d[lons2d > 180] -= 360

    x, y = basemap(lons2d, lats2d)
    #x = (x[1:,1:] + x[:-1, :-1]) /2.0


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3)

    plot_utils.apply_plot_params(width_pt=None, width_cm=25,height_cm=35, font_size=16)
    fig = plt.figure()
    assert isinstance(fig, Figure)


    h_max = 10
    cmap = cm.get_cmap("cool",10)
    cmap.set_over(cmap(1.0))
    clevels = np.arange(0,h_max+1,1)
    gs = gridspec.GridSpec(3,1)

    all_axes = []
    all_img = []


    i = 0
    for name in sim_names:
        path = simname_to_path[name]
        dm = CRCMDataManager(data_folder=path)
        hc = dm.get_alt_using_monthly_mean_climatology(xrange(start_year,end_year+1))
        ax = fig.add_subplot(gs[i+1,0])
        assert isinstance(ax, Axes)
        hc = np.ma.masked_where(mask_cond, hc)
        img = basemap.pcolormesh(x, y, hc, cmap = cmap, vmax = h_max)
        if not i:
            ax.set_title("ALT from climatology ({0} - {1})".format(start_year, end_year))
        i += 1
        ax.set_ylabel(name)
        all_axes.append(ax)
        all_img.append(img)



    i = 0
    axs_to_hide = []
    #zones and coastlines
    for the_ax, the_img in zip(all_axes, all_img):
        divider = make_axes_locatable(the_ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = fig.colorbar(the_img,  cax = cax, extend = "max")

        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=0.5)
        basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
                ax=the_ax, linewidth=1.5)

        if i:
            axs_to_hide.append(cax)
        i += 1


    #draw zones
    ax = fig.add_subplot(gs[0,0])
    basemap.drawcoastlines(ax = ax, linewidth=1.5)
    shp_info = basemap.readshapefile("data/pf_2/permafrost5_wgs84/permaice", name="zone",
            ax=ax, linewidth=3)
    print shp_info
    for nshape,seg in enumerate(basemap.zone):
        the_color = "green" if basemap.zone_info[nshape]["EXTENT"] == "C" else "red"
        poly = mpl.patches.Polygon(seg,facecolor=the_color, zorder = 10)
        ax.add_patch(poly)

    b1 = ax.bar([0],[0],color="green", label="Continuous")
    b2 = ax.bar([0],[0],color="r", label="Discontinuous")
    ax.legend(loc = "lower left")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(all_img[0],  cax = cax, extend = "max")
    axs_to_hide.append(cax)

    fig.tight_layout()

    for the_ax in axs_to_hide:
        the_ax.set_visible(False)

    fig.savefig("alt_from_climatology_future.png")






def plot_current_alts():

    import plot_dpth_to_bdrck
    bdrck_field = plot_dpth_to_bdrck.get_depth_to_bedrock()
    start_year = 1980
    end_year = 1996

    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"
    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    #coord_file = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NA_1.0deg_soil_spinup2/pmNA_1.0deg_soil_spinup2_228006_moyenne"

    sim_names = ["ERA40", "MPI","CanESM"]
    simname_to_path = {
        #"ERA40": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1",
        "ERA40": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_ERA40-Int_old_snow_cond",
        #"ERA40" : "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NA_1.0deg_soil_spinup2",
        "MPI": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_MPI_B1",
        "CanESM": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1"

    }


    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=45.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74,
        anchor="W"
    )
    assert isinstance(basemap, Basemap)

    #basemap.transform_scalar()

    #basemap = Basemap()
    lons2d[lons2d > 180] -= 360

    x, y = basemap(lons2d, lats2d)
    #x = (x[1:,1:] + x[:-1, :-1]) /2.0


    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3)

#    plot_utils.apply_plot_params(width_pt=None, width_cm=20, height_cm=40, font_size=16)
    fig = plt.figure()
    assert isinstance(fig, Figure)


    h_max = 10
    bounds = [0,0.1,0.5,1,2,3,4,5,6]
    cmap = my_colormaps.get_lighter_jet_cmap(ncolors=len(bounds) - 1) #cm.get_cmap("jet",10)
    norm = BoundaryNorm(boundaries=bounds,ncolors=len(bounds), clip=True)
    cmap.set_over(cmap(1.0))
    clevels = np.arange(0,h_max+1,1)
    gs = gridspec.GridSpec(1,2, width_ratios=[1,0.1], hspace=0, wspace=0.1,
        left=0.05, bottom = 0.02, top=0.95)

    all_axes = []
    all_img = []


    i = 0
    hc_list = []

    for name in sim_names:
        path = simname_to_path[name]
        dm = CRCMDataManager(data_folder=path)
        hc0, t3d_min, t3d_max = dm.get_alt_using_monthly_mean_climatology(xrange(start_year,end_year+1))

        hc_list.append(hc0)
        ax = fig.add_subplot(gs[i,0])
        cp = SoundingPlotter(ax, basemap, t3d_min, t3d_max, lons2d, lats2d, levelheights=dm.level_heights)

        assert isinstance(ax, Axes)
        hc = np.ma.masked_where(mask_cond | (hc0 < 0), hc0)
        #hc = np.ma.masked_where( (hc0 < 0), hc0)
        hc5 = np.ma.masked_where((hc0 <= 15) | hc.mask, hc)
        img = basemap.pcolormesh(x, y, hc, cmap = cmap, vmax = h_max, norm=norm)
        if not i:
            ax.set_title("ALT ({0} - {1}) \n".format(start_year, end_year))
        i += 1
        ax.set_ylabel("CRCM ({0})".format(name))
        all_axes.append(ax)
        all_img.append(img)
        print np.ma.min(hc), np.ma.max(hc)
        #hc5 = np.ma.masked_where((hc0 <= 6) | hc.mask, hc)
        #print "Number of cells with alt > 5 is {0}, and the range is {1} ... {2}".format(hc5.count(), hc5.min(), hc5.max())
        #bdrck_field5 = np.ma.masked_where(hc5.mask, bdrck_field)
        #print "Bedrock ranges for those points: {0} ... {1}".format(bdrck_field5.min(), bdrck_field5.max())


        ind = np.where(~hc5.mask)
        xs = ind[0]
        ys = ind[1]

        all_months, all_temps = dm.get_monthly_mean_soiltemps(year_range=xrange(start_year,end_year+1))
        all_months_ord = date2num(all_months)
        i = 0
        mpl.rcParams['contour.negative_linestyle'] = 'solid'

        for the_i, the_j in zip(xs,ys):
            #plot profile
            plt.figure()
            plt.plot(t3d_max[the_i, the_j, :] - dm.T0, dm.level_heights, color = "r")
            plt.plot(t3d_min[the_i, the_j, :] - dm.T0, dm.level_heights, color = "b")
            plt.plot([0 , 0], [dm.level_heights[0], dm.level_heights[-1]], color = "k")

            x1, x2 = plt.xlim()
            #plt.plot( [x1, x2], [bdrck_field[the_i, the_j], bdrck_field[the_i, the_j]], color = "k", lw = 3 )
            #plt.title(str(i) + ", dpth_to_bedrock = {0} m".format(bdrck_field[the_i, the_j]))
            plt.title(str(i))
            plt.gca().invert_yaxis()
            plt.savefig("prof{0}.png".format(i))
            ax.annotate(str(i), (x[the_i, the_j], y[the_i, the_j]), font_properties =
                            FontProperties(size=10))


            #plot vertical temp cross-section
            plt.figure()
            plt.title(str(i) + ", ({0} - {1})".format(start_year, end_year))

            levs2d, times2d = np.meshgrid(dm.level_heights, all_months_ord)
            clevs = [-25,-20,-10,-5,-1,0,1,5,10,20,25]
            norm = BoundaryNorm(boundaries=clevs, ncolors=len(clevs) - 1)
            cmap = cm.get_cmap("jet", len(clevs) - 1)

            img = plt.contourf(times2d, levs2d, all_temps[:,the_i, the_j, :] - dm.T0, levels = clevs, cmap = cmap, norm = norm)
            #plt.contour(times2d, levs2d, all_temps[:,the_i, the_j, :] - dm.T0, levels = clevs, colors = "k", linewidth = 1)
            the_ax = plt.gca()
            assert isinstance(the_ax, Axes)
            the_ax.invert_yaxis()
            the_ax.xaxis.set_major_formatter(FuncFormatter(
                lambda x, pos: num2date(float(x)).strftime("%Y")
            ))

            print "i = {0}; lon, lat = {1}, {2}".format(i, lons2d[the_i, the_j], lats2d[the_i, the_j])

            plt.colorbar(img, ticks = clevs)

            plt.savefig("temp_section_{0}.png".format(i))

            i += 1

        print  "lons = [{0}]".format(",".join(map( lambda x: str(x), lons2d[np.array(xs), np.array(ys)])))
        print  "lats = [{0}]".format(",".join(map( lambda x: str(x), lats2d[np.array(xs), np.array(ys)])))




        plt.figure()
        alt_ranges = xrange(0,18)
        numbers = []
        for the_alt in alt_ranges:
            hci = np.ma.masked_where( (hc0 < the_alt) | (hc0 > the_alt + 1) | hc.mask ,hc)
            numbers.append(hci.count())
        plt.bar(alt_ranges, numbers, width=1)
        plt.xlabel("ALT (m)")
        plt.ylabel("Number of cells")
        plt.savefig("numbers_in_range.png")

        break



    i = 0
    axs_to_hide = []
    #zones and coastlines
    for the_ax, the_img in zip(all_axes, all_img):
#        divider = make_axes_locatable(the_ax)
#        cax = divider.append_axes("right", "5%", pad="3%")
        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=0.5)
        basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
                ax=the_ax, linewidth=1.5, drawbounds=False)

        for nshape,seg in enumerate(basemap.zone):
            if basemap.zone_info[nshape]["EXTENT"] not in  ["C","D"]: continue
            poly = mpl.patches.Polygon(seg,edgecolor = "k", facecolor="none", zorder = 10, lw = 1.5)
            the_ax.add_patch(poly)


#        if i != 1:
#            axs_to_hide.append(cax)
        i += 1

    cax = fig.add_subplot(gs[:,1])
    cax.set_anchor("W")
    cax.set_aspect(30)
    formatter = FuncFormatter(
        lambda x, pos: "{0: <6}".format(x)
    )
    cb = fig.colorbar(all_img[0], ax = cax, cax = cax, extend = "max", ticks = bounds, format = formatter)

    cax.set_title("m")


    #fig.tight_layout(h_pad=0)

#    for the_ax in axs_to_hide:
#        the_ax.set_visible(False)

    fig.savefig("alt_from_climatology_current.png")

    #print ALT for selected points
    site_names = ["S","K","T"]
    sel_lons = [-75.646, -65.92, -69.95]
    sel_lats = [62.197, 58.709, 58.67]

    xo,yo,zo = lat_lon.lon_lat_to_cartesian(sel_lons, sel_lats)

    xi, yi, zi = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
    ktree = KDTree(zip(xi,yi,zi))
    dists, indexes =  ktree.query(zip(xo,yo,zo))

    for name, data in zip(sim_names, hc_list):
        print name
        flat_data = data.flatten()
        for p_name, ind in zip(site_names, indexes):
            print p_name, "{0} m".format(flat_data[ind])
        print "--" * 10


    pass

if __name__ == "__main__":
    import application_properties
    plot_utils.apply_plot_params(width_pt=None, width_cm=28, height_cm=20, font_size=20)
    application_properties.set_current_directory()
    #plot_future_alts()
#    plot_current_alts_nyear_rule()
    plot_current_alts()
#    plot_mean_alt_from_jpp_results()
    #plt.show()
    #main()
    print "Hello world"
  
