import osgeo.ogr
__author__="huziy"
__date__ ="$12 nov. 2010 19:19:48$"


from osgeo import ogr
from osgeo import osr


import numpy as np

from matplotlib.patches import Polygon
from shapely.wkt import loads
#import mapscript

import application_properties
application_properties.set_current_directory()

def get_features_from_shape(basemap, path = 'data/shape/contour_bv_MRCC/Bassins_MRCC_utm18.shp', linewidth = 2,
                        edgecolor = 'k'):
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


        #get fields of the feature
#        for i in xrange(feature.GetFieldCount()):
#            print feature.GetField(i)

#        print geom.ExportToWkt()

        polygon = loads(geom.ExportToWkt())
        boundary = polygon.exterior
        coords = np.zeros(( len(boundary.coords), 2))
        for i, the_coord in enumerate(boundary.coords):


#            if feature.GetFieldAsString('abr').lower() == 'rdo':
#                print the_coord[0], the_coord[1]
            if basemap is not None:
                coords[i, 0], coords[i, 1] = basemap( the_coord[0], the_coord[1] )

        result.append(Polygon(coords, facecolor = 'none', linewidth = linewidth, edgecolor = edgecolor))
        feature = layer.GetNextFeature()


    dataStore.Destroy()
    return result



def get_copies(patches):
    result = []
    for p in patches:
        result.append(Polygon(p.get_xy(), facecolor = p.get_facecolor(),
                      edgecolor = p.get_edgecolor(), linewidth = p.get_linewidth()))
    return result


def reproject_to_latlon_and_save_shape(path = 'data/shape/contour_bv_MRCC/Bassins_MRCC_utm18.shp',
                                       path_new = 'data/shape/contour_bv_MRCC/Bassins_MRCC_latlon.shp'):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(path, 0)
    layer = dataStore.GetLayer(0)
    latlong = osr.SpatialReference()
    latlong.ImportFromProj4("+proj=latlong")

    print latlong

    shapeData = driver.CreateDataSource(path_new)
    newLayer = shapeData.CreateLayer('basin_boundaries', latlong, osgeo.ogr.wkbPolygon)


    #project geometries of the features
    feature = layer.GetNextFeature()
    while feature:
        geom = feature.GetGeometryRef()
        geom.TransformTo(latlong)
        newFeature = ogr.Feature(newLayer.GetLayerDefn())
        newFeature.SetGeometry(geom)
        newLayer.CreateFeature(feature)
        feature = layer.GetNextFeature()

    shapeData.Destroy()

    pass
    

if __name__ == "__main__":
    get_features_from_shape(None)
#    reproject_to_latlon_and_save_shape()
    print "Hello World"
