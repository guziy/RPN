import osgeo
from shapely.geometry.linestring import LineString

__author__ = 'huziy'

import numpy as np

from osgeo import ogr, osr
from shapely.geometry import Polygon, MultiLineString
from shapely import wkt




def get_qc_rivers(path = "data/shp/rivers_qc/Riviere_QC_utm18.shp"):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(path, 0)
    layer = dataStore.GetLayer(0)
    assert isinstance(layer, ogr.Layer)


    latlong = osr.SpatialReference()
    latlong.ImportFromProj4("+proj=latlong")
    result = []

    assert isinstance(dataStore, ogr.DataSource)
    feature = layer.GetNextFeature()
    while feature:
        geom = feature.GetGeometryRef()
        geom.TransformTo(latlong)
        assert isinstance(geom, ogr.Geometry)
        river = wkt.loads( geom.ExportToWkt() )

        result.append( river )
        feature = layer.GetNextFeature()

    dataStore.Destroy()
    return result





def get_rivers_in_region(lon_min, lon_max, lat_min, lat_max, path = ""):
    """
    This method is for small regions, not intersecting pm 180 degrre meridian
    """
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(path, 0)
    layer = dataStore.GetLayer(0)
    assert isinstance(layer, ogr.Layer)


    latlong = osr.SpatialReference()
    latlong.ImportFromProj4("+proj=latlong")
    result = []

    assert isinstance(dataStore, ogr.DataSource)

    shell = [
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_max),
            (lon_max, lat_min),
            (lon_min, lat_min)
    ]

    poly = Polygon(shell=shell)

    g = ogr.CreateGeometryFromWkt(poly.wkt)
    assert isinstance(g, ogr.Geometry)
    layer.SetSpatialFilter(g)

    feature = layer.GetNextFeature()
    while feature:
        geom = feature.GetGeometryRef()
        assert isinstance(geom, ogr.Geometry)
        result.append( wkt.loads( geom.ExportToWkt() ) )
        feature = layer.GetNextFeature()

    dataStore.Destroy()
    return result


def reproject_rivers_to_latlon_and_save_shape(path = 'data/shp/rivers_qc/Riviere_QC_utm18.shp',
                                       path_new = 'data/shp/rivers_qc_latlon/qc_rivs_latlon.shp'):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    dataStore = driver.Open(path, 0)
    layer = dataStore.GetLayer(0)
    latlong = osr.SpatialReference()
    latlong.ImportFromProj4("+proj=latlong")

    print latlong

    shapeData = driver.CreateDataSource(path_new)
    newLayer = shapeData.CreateLayer('qc_rivers', latlong, osgeo.ogr.wkbLineString)


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


def main():
    import application_properties
    application_properties.set_current_directory()
    reproject_rivers_to_latlon_and_save_shape()
    #get_qc_rivers()

    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  