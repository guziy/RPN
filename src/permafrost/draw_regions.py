from netCDF4 import Dataset
import os
import pickle
import itertools
from matplotlib.colors import ListedColormap
from matplotlib.ticker import  MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap, maskoceans

__author__ = 'huziy'

import numpy as np
import application_properties


from osgeo import ogr
from osgeo import osr

from rpn.rpn import RPN
import matplotlib.pyplot as plt

#from descartes.patch import PolygonPatch

permafrost_types = ("C", "D", "S", "I")
permafrost_types_long_names = ("Continuous", "Discontinuous", "Sporadic", "Isolated")



def delete_points_in_countries(points_lat_long, points, indices, countries = None,
                               path = "data/shp/countries/cntry00.shp"):


    query = "CNTRY_NAME IN (\'" + "\',\'".join(countries) + "\')"
    print query
    ogr.UseExceptions()
    osr.UseExceptions()
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(path, 0)
    layer = dataStore.GetLayer(0)
    layer.SetAttributeFilter(query)



    feature = layer.GetNextFeature()
    while feature:
        print feature.items()["CNTRY_NAME"]
        geom = feature.GetGeometryRef()

        if feature.items()["CNTRY_NAME"] != "Russia":
            print geom.ExportToWkt()


        to_del = itertools.ifilter(lambda x: geom.Distance(x[1]) < 0.2 ,
                                        zip(points, points_lat_long, indices))

        to_del = list(to_del)
        for p_del, p_ll_del, i_del in to_del:
            points.remove(p_del)
            indices.remove(i_del)
            points_lat_long.remove(p_ll_del)
        feature = layer.GetNextFeature()


        pass



    dataStore.Destroy()


def get_basemap_and_coords(
        file_path = "data/CORDEX/NorthAmerica_0.44deg_CanHistoE1/Samples/NorthAmerica_0.44deg_CanHistoE1_198101/pm1950010100_00816912p",
        lon1 = -97, lat1 = 47.50,
        lon2 = -7, lat2 = 0,
        llcrnrlon = None, llcrnrlat = None,
        urcrnrlon = None, urcrnrlat = None, resolution = "l"
        ):
    rpnObj = RPN(file_path)
    lons2D, lats2D = rpnObj.get_longitudes_and_latitudes()
    rpnObj.close()


    the_ll_lon = lons2D[0,0] if llcrnrlon is None else llcrnrlon
    the_ll_lat = lats2D[0,0] if llcrnrlat is None else llcrnrlat
    the_ur_lon = lons2D[-1, -1] if urcrnrlon is None else urcrnrlon
    the_ur_lat = lats2D[-1, -1] if urcrnrlat is None else urcrnrlat

    return Basemap(projection="omerc", resolution=resolution,
            llcrnrlon=the_ll_lon,
            llcrnrlat=the_ll_lat,
            urcrnrlon=the_ur_lon,
            urcrnrlat=the_ur_lat,
            lat_1=lat1, lon_1=lon1, lat_2=lat2, lon_2=lon2, no_rot=True
    ), lons2D, lats2D

    pass


def create_gdal_point_and_transform(x, y, transformation = None):
    p = ogr.CreateGeometryFromWkt("POINT(%f %f)" % (x, y))
    if transformation is not None:
        p.Transform(transformation)
    return p

def create_points_envelope_gdal(points):

    xs = map(lambda x: x.GetX(), points)
    xmin = min(xs)
    xmax = max(xs)

    ys = map(lambda x: x.GetY(), points)
    ymin = min(ys)
    ymax = max(ys)

    rect_s = "POLYGON ((%f %f,%f %f, %f %f ,%f %f,%f %f))"
    rect_s = rect_s % (xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin)
    return ogr.CreateGeometryFromWkt(rect_s)
    pass

def get_permafrost_mask(lons2d, lats2d, zones_path = "data/permafrost/permaice.shp"


                        ):

    cache_file = "permafrost_types.bin"

    if os.path.isfile(cache_file):
       return pickle.load(open(cache_file))



    ogr.UseExceptions()


    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(zones_path, 0)
    layer = dataStore.GetLayer(0)



    latlong = osr.SpatialReference()
    latlong.ImportFromProj4("+proj=latlong")

    ct = osr.CoordinateTransformation(latlong, layer.GetSpatialRef())



    points = map(lambda x:  create_gdal_point_and_transform(x[0], x[1], ct),
        zip(lons2d.flatten(), lats2d.flatten()))

    points_lat_long = map(lambda x:  create_gdal_point_and_transform(x[0], x[1]),
            zip(lons2d.flatten(), lats2d.flatten()))

    i_indices_1d = np.array(xrange(lons2d.shape[0]))
    j_indices_1d = np.array(xrange(lons2d.shape[1]))


    j_indices_2d, i_indices_2d = np.meshgrid(j_indices_1d, i_indices_1d)
    indices = zip(i_indices_2d.flatten(), j_indices_2d.flatten())
    indices = list(indices)

    ##do not consider territories of the following countries
    rej_countries = ["Greenland", "Iceland", "Russia"]
    delete_points_in_countries(points_lat_long, points, indices, countries = rej_countries)

    permafrost_kind_field = np.zeros(lons2d.shape)
    grid_polygon = create_points_envelope_gdal(points)

    #set spatial and attribute filters to take only the features with valid EXTENT field,
    #and those which are close to the area of interest
    layer.SetSpatialFilter(grid_polygon)
    query = "EXTENT IN  (\'{0}\',\'{1}\',\'{2}\' ,\'{3}\')".format(*permafrost_types)
    query += "OR EXTENT IN  (\'{0}\',\'{1}\',\'{2}\' ,\'{3}\')".format(*map(lambda x: x.lower(), permafrost_types))
    print query
    layer.SetAttributeFilter(query)



    print layer.GetFeatureCount()
    print grid_polygon.ExportToWkt()

    ##read features from the shape file
    feature = layer.GetNextFeature()
    i = 0
    while feature:
        geom = feature.GetGeometryRef()
        points_to_remove = []
        indices_to_remove = []
        for ind, p in zip(indices, points):
            if geom.Intersect(p):
                perm_type = feature.items()["EXTENT"]
                permafrost_kind_field[ind] = permafrost_types.index(perm_type) + 1
                points_to_remove.append(p)
                indices_to_remove.append(ind)
                print i

        for the_p, the_i in zip(points_to_remove, indices_to_remove):
            indices.remove(the_i)
            points.remove(the_p)

        feature = layer.GetNextFeature()

        i += 1

    dataStore.Destroy()
    pickle.dump(permafrost_kind_field, open(cache_file, "w"))
    return permafrost_kind_field


def main():
    figure = plt.figure()
    basemap, lons2d, lats2d = get_basemap_and_coords()

    lons2d[lons2d >180] -= 360


    x0, y0 = basemap(lons2d, lats2d)
    dx = x0[1,0] - x0[0,0]
    dy = y0[0,1] - y0[0,0]
    x1 = x0 - dx / 2.0
    y1 = y0 - dy / 2.0

    permafrost_kind_field = get_permafrost_mask(lons2d, lats2d)

    cmap = ListedColormap(["r", "b", "y", "c"])
    cmap.set_over("w")
    cmap.set_under("w")

    #permafrost_kind_field = np.ma.masked_where(permafrost_kind_field == 0, permafrost_kind_field)

    ax_map = plt.gca()
    #img = basemap.pcolormesh(x1, y1, permafrost_kind_field, ax = ax_map, vmin = 0.5, vmax = 4.5, cmap = cmap )
    permafrost_kind_field = maskoceans(lons2d, lats2d, permafrost_kind_field)
    
    img = basemap.contourf(x0, y0, permafrost_kind_field, levels = np.arange(0.5, 5, 0.5), cmap = cmap)

    divider = make_axes_locatable(ax_map)
    cax = divider.append_axes("bottom", "5%", pad="3%")
    cb = plt.colorbar(img, ticks = MultipleLocator(), cax = cax, orientation = "horizontal")

    

    basemap.contour(x0, y0, permafrost_kind_field, ax = ax_map,
                        levels = range(6), linewidths=0.5, colors = "k")
    basemap.drawcoastlines(ax = ax_map)
    plt.savefig("test.png")



    #gdal.Dataset.
    #TODO: implement
    pass


def get_EASE_basemap():
    b = Basemap(projection="nplaea",lon_0=180, lat_0=90, boundinglat=25)
    print b.proj4string
    return b


def test_ease_basemap():
    b = get_EASE_basemap()
    b.drawcoastlines()
    plt.savefig("ease_region.png")


def save_pf_mask_to_netcdf(path = "permafrost_types.nc"):
    ds = Dataset(path, mode = "w", format="NETCDF3_CLASSIC")

    b, lons2d, lats2d = get_basemap_and_coords()
    pf_mask = get_permafrost_mask(lons2d, lats2d)
    ds.createDimension('lon', lons2d.shape[0])
    ds.createDimension('lat', lons2d.shape[1])

    lonVariable = ds.createVariable('longitude', 'f4', ('lon', 'lat'))
    latVariable = ds.createVariable('latitude', 'f4', ('lon', 'lat'))
    maskVariable = ds.createVariable("pf_type", "i4", ('lon', 'lat'))
    maskVariable.description = "0-no data, 1-Continuous, 2-Discontinuous, 3-Sporadic, 4-Isolated"

    maskVariable[:,:] = pf_mask[:,:]
    lonVariable[:,:] = lons2d[:,:]
    latVariable[:,:] = lats2d[:,:]

    ds.close()

if __name__ == "__main__":
    application_properties.set_current_directory()
    #test_ease_basemap()
    #main()
    save_pf_mask_to_netcdf()
    print "Hello world"
  