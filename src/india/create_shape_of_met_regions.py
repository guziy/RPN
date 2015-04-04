import os
import sys

__author__ = 'huziy'

import numpy as np
from xml.dom.minidom import parse, Element
from osgeo import ogr
from osgeo import osr
from osgeo import gdal


def get_screen_coords_of_shapes():
    path = "data/india_met_regions/met_polygons.txt"
    dom = parse(path)

    area_els = dom.getElementsByTagName("area")
    a = area_els[0]
    assert isinstance(a, Element)
    return [x.getAttribute("coords") for x in area_els]

def transform_to_lon_lat(x, y, coefs):
    a, b, c, d, e, f = coefs
    lon = a * x + b * y + c
    lat = d * x + e * y + f
    return lon, lat


def get_transform_params():
    lons = [
        79 + 52.0/60.0 + 21.77/3600.0,
        77 + 32.0/60.0 + 39.68/3600.0,
        69 + 52.0/60.0 + 21.77/3600.0
    ]
    lats = [
        10 + 17.0/60.0 + 56.87/3600.0,
        8 + 4.0/60.0 + 48.47/3600.0,
        22 + 26.0/60.0 + 39.81/3600.0,
    ]


    xs = [344.5, 269.0, 72.0]
    ys = [834.0, 884.0, 500.5]

    #ys = [938 - y + 1 for y in ys]
    

    matrix = [
        [xs[0], ys[0],1,0,0,0],
        [0,0,0,xs[0], ys[0], 1],

        [xs[1], ys[1],1,0,0,0],
        [0,0,0,xs[1], ys[1], 1],

        [xs[2], ys[2],1,0,0,0],
        [0,0,0,xs[2], ys[2], 1]
    ]
    
    rhs = [lons[0], lats[0], lons[1], lats[1], lons[2], lats[2]]

    coefs = np.linalg.solve(matrix, rhs)
    return coefs


def main():

    coords_lists = get_screen_coords_of_shapes()


    coefs = get_transform_params()
    point_lists = []

    driverName = "ESRI Shapefile"
    drv = ogr.GetDriverByName( driverName )
    if drv is None:
        print("%s driver not available.\n" % driverName)
        sys.exit( 1 )
    shape_file_name = "india_met_regions.shp"
    if os.path.isfile(shape_file_name):
        os.remove(shape_file_name)
    ds = drv.CreateDataSource( shape_file_name )
    print(ds)
    assert isinstance(ds, ogr.DataSource)
    if ds is None:
        print("Creation of output file failed.\n")
        sys.exit( 1 )

    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")

    lyr = ds.CreateLayer( "regions", srs, ogr.wkbPolygon )
    assert isinstance(lyr, ogr.Layer)
    if lyr is None:
        print("Layer creation failed.\n")
        sys.exit( 1 )


    x = None
    y = None
    for cl in coords_lists:
        fields = cl.split(",")
        fields = [float(f.strip()) for f in fields]
        region = ogr.Geometry(ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for i, f in enumerate( fields ):
            if not i % 2:
                x = f
            else:
                y = f
                lon, lat = transform_to_lon_lat(x, y, coefs)
                ring.AddPoint(lon, lat)

        feat = ogr.Feature(lyr.GetLayerDefn())
        ring.CloseRings()
        region.AddGeometry(ring)

        feat.SetGeometry(region)
        if lyr.CreateFeature(feat):
            print("Failed to create feature in shapefile.\n")
            sys.exit( 1 )

        feat.Destroy()
    ds.Destroy()


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print("Hello world")
  
