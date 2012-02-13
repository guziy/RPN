# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="huziy"
__date__ ="$25 fevr. 2011 14:43:55$"



from osgeo import ogr
from osgeo import osr


import numpy as np

from matplotlib.patches import Polygon
from shapely.wkt import loads


def test():
    path = 'contour_bv_MRCC/Bassins_MRCC_utm18.shp'
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

        polygon = loads(geom.ExportToWkt())
        boundary = polygon.exterior
        coords = np.zeros(( len(boundary.coords), 2))

        result.append(Polygon(coords, facecolor = 'none', linewidth = 1))
        feature = layer.GetNextFeature()


    dataStore.Destroy()
    return result

    pass

if __name__ == "__main__":
    test()
    print "Hello World"
