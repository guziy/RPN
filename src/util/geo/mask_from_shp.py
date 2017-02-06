from pathlib import Path

from application_properties import main_decorator
from osgeo import ogr
import numpy as np


import hashlib
import pickle

mask_cache_folder = Path("mask_caches")

def get_cache_file_path(lons2d, lats2d, shp_path="", polygon_name=None, hints=None):

    if not mask_cache_folder.exists():
        mask_cache_folder.mkdir()

    coord_tuple = (lons2d.min(), lons2d.max(), lons2d.mean(), lons2d.std(),
                   lats2d.min(), lats2d.max(), lats2d.mean(), lats2d.std(),)


    res_tuple = coord_tuple + (shp_path, )

    if polygon_name is not None:
        res_tuple += (polygon_name, )

    if hints is not None:
        for fieldname in sorted(hints):
            res_tuple += (fieldname, hints[fieldname])


    res_tuple = (str(el) for el in res_tuple)

    file_name = hashlib.sha224("".join(list(res_tuple)).encode()).hexdigest()


    file_path = mask_cache_folder.joinpath("{}.bin".format(file_name))

    return file_path



def does_layer_has_att(layer: ogr.Layer, att_name: str):
    """

    :param layer: layer to look for attributes in
    :param att_name: the name of the attribute to find
    :return:
    """
    feature_defn = layer.GetLayerDefn()
    """
    :type feature_defn: ogr.FeatureDefn
    """

    for field_index in range(feature_defn.GetFieldCount()):
        field_defn = feature_defn.GetFieldDefn()
        """
        :type field_defn: ogr.FieldDefn
        """
        if field_defn.GetName() == att_name:
            return True

    return False



def get_mask(lons2d, lats2d, shp_path="", polygon_name=None, hints=None):
    """
    Assumes that the shape file contains polygons in lat lon coordinates
    :param hints: a dict of {fieldname: fieldvalue} for the polygons, if any of the hints correspond to a polygon, the polygon is treated
    :param lons2d:
    :param lats2d:
    :param shp_path:
    :rtype : np.ndarray
    The mask is >= 1 for the points inside of the polygons
    """

    assert Path(shp_path).exists()


    cache = get_cache_file_path(lons2d=lons2d, lats2d=lats2d, shp_path=shp_path, polygon_name=polygon_name, hints=hints)

    if cache.exists():
        return pickle.load(cache.open(mode="rb"))




    ds = ogr.Open(shp_path)
    """
    :type : ogr.DataSource
    """

    xx = lons2d.copy()
    yy = lats2d

    # set longitudes to be from -180 to 180
    xx[xx > 180] -= 360

    mask = np.zeros(lons2d.shape, dtype=int)
    nx, ny = mask.shape

    pt = ogr.Geometry(ogr.wkbPoint)

    feature_id = 1
    for i in range(ds.GetLayerCount()):
        layer = ds.GetLayer(i)
        """
        :type : ogr.Layer
        """


        if polygon_name is not None:
            # check if the file has the name atribute
            if does_layer_has_att(layer, "name"):
                layer.SetAttributeFilter("name = '{}'".format(polygon_name))


        if hints is not None:
            filters = []
            for field_name, field_value in hints.items():
                if does_layer_has_att(layer, field_name):
                    filters.append("{} = '{}'".format(field_name, field_value))


            print("Attribute filter: {}".format(" or ".join(filters)))
            layer.SetAttributeFilter(" or ".join(filters))


        print(layer.GetFeatureCount())

        feat = layer.GetNextFeature()
        while feat is not None:
            """
            :type : ogr.Feature
            """


            # for att_i in range(feat.GetFieldCount()):
            #     field_defn = feat.GetFieldDefnRef(att_i)
            #     """
            #     :type field_defn: ogr.FieldDefn
            #     """
            #     print("{} = {}".format(field_defn.GetName(), feat.GetField(att_i)))




            g = feat.GetGeometryRef()

            """
            :type : ogr.Geometry
            """


            # assert isinstance(g, ogr.Geometry)

            for pi in range(nx):
                for pj in range(ny):
                    pt.SetPoint_2D(0, float(xx[pi, pj]), float(yy[pi, pj]))


                    if g.Contains(pt):
                        mask[pi, pj] += feature_id

            feature_id += 1

            feat = layer.GetNextFeature()



    pickle.dump(mask, cache.open(mode="wb"))
    return mask


@main_decorator
def main():
    pass


if __name__ == '__main__':
    main()
