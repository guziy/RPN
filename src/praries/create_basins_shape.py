import os
from osgeo.ogr import Layer
from shapely.geometry.point import Point

__author__ = 'huziy'

import numpy as np
from osgeo import ogr
from osgeo import osr
from shapely.geometry import Polygon
from domains.map_parameters_amno import polar_stereographic
import shutil

def print_indices(polygons, lons2d, lats2d):

    pass

def main():

    spatialReference = osr.SpatialReference()
    spatialReference.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')


    driver = ogr.GetDriverByName("ESRI Shapefile")

    shp_dir = "praries_basins"
    shutil.rmtree(shp_dir)
    shapeData = driver.CreateDataSource(shp_dir)



    layer = shapeData.CreateLayer('layer1', spatialReference, ogr.wkbPolygon)
    #assert isinstance(layer, Layer)
    layer.CreateField(ogr.FieldDefn("BasinId"))
    layerDefinition = layer.GetLayerDefn()


    input_path = "data/praries_basins/Boundaries_lat_lon.txt"

    lines = open(input_path).readlines()


    lons2d = polar_stereographic.lons
    lats2d = polar_stereographic.lats

    id_to_points = {}
    anomaly_i = 1
    point_prev = None
    for line in lines:
        if line.strip() == "": continue
        fields = line.split()
        the_id = fields[0]
        lon = float(fields[1])
        lat = float(fields[2])

        if not id_to_points.has_key(the_id):
            id_to_points[the_id] = []

        if int(the_id) >= 16:
            the_point = Point(lon, lat)
            if point_prev is not None:
                the_dist = the_point.distance(point_prev)
                if the_dist > 0.5:
                    id_to_points[the_id + "_{0}".format(anomaly_i)] = id_to_points[the_id]
                    id_to_points[the_id] = []
                    anomaly_i += 1
            point_prev = the_point

        id_to_points[the_id].append((lon, lat))
        pass

    polygons = []
    featureIndex = 0
    for the_id, points in id_to_points.iteritems():

        if not len(points): continue

        feature = ogr.Feature(layerDefinition)

        feature.SetField("BasinId", the_id)

        #create a polygon using shapely
        p = Polygon(shell=points)
        #print p.wkt
        polygons.append(p)
        pGdal = ogr.CreateGeometryFromWkb(p.wkb)
        feature.SetGeometry(pGdal)
        feature.SetFID(featureIndex)
        layer.CreateFeature(feature)
        featureIndex += 1

    shapeData.Destroy()




    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  