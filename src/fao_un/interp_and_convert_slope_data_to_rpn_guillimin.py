import os
from netCDF4 import Dataset
import geopy
from geopy import distance
import numpy as np
from rpn.rpn import RPN
from scipy import sparse
from scipy.spatial.ckdtree import cKDTree
from util.geo import lat_lon

__author__ = 'huziy'

import pandas as pd

SLOPE_CLASS_TO_MEDIAN = {
    1: 0.25 * 1e-2,
    2: 1.25 * 1e-2,
    3: 3.50 * 1e-2,
    4: 7.50 * 1e-2,
    5: 12.5 * 1e-2,
    6: 22.5 * 1e-2,
    7: 37.5 * 1e-2,
    8: 52.5 * 1e-2
}


def _get_source_lon_lat(path=""):
    params = {}
    with open(path) as f:
        for i in range(6):
            key, val = f.readline().strip().split()
            val = float(val)
            params[key] = val

    params["nrows"] = int(np.round(params["nrows"]))
    params["ncols"] = int(np.round(params["ncols"]))

    d = params["cellsize"]
    lon1d = [params["xllcorner"] + i * d for i in range(params["ncols"])]

    lat1d = [params["yllcorner"] + i * d for i in range(params["nrows"])]
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    return params, lon2d, lat2d


def _get_slope_data(path=""):
    """
    cols - the list of column indices to be read
    """
    data = np.loadtxt(path, skiprows=6, dtype=np.uint8)
    return np.flipud(data)


def interpolate_slopes(in_path_template="",
                       in_path_rpn_geophy="/skynet3_rech1/huziy/geof_lake_infl_exp/geophys_Quebec_0.1deg_260x260_with_dd_v6",
                       out_path_rpn_geophy=None, var_name_with_target_coords="Z0"):
    """
    interpolate slope data at in_path_template (template because there are 8 files, 1 for each class),
    to a grid defined in a geophy file (rpn)
    Result: an rpn file containing interflow slopes field
    :param in_path_template:
    :param in_path_rpn_geophy:
    :param out_path_rpn_geophy:
    """
    if out_path_rpn_geophy is None:
        out_path_rpn_geophy = in_path_rpn_geophy + "_with_ITFS"

    r_obj_in = RPN(in_path_rpn_geophy)
    r_obj_out = RPN(out_path_rpn_geophy, mode="w")

    data = []
    i = 0

    npas_for_sl = None
    deet_for_sl = None
    dateo_for_sl = None
    ips_for_sl = None
    igs_for_sl = None
    nbits_for_sl = None
    data_type_for_sl = None
    typ_var_for_sl = None
    grid_type_for_sl = None


    while data is not None:
        data = r_obj_in.get_next_record()
        if data is None:
            break
        info = r_obj_in.get_current_info()

        nbits = info["nbits"]
        data_type = info["data_type"]

        if nbits > 0:
            nbits = -nbits

        ips = info["ip"]

        npas = info["npas"]
        deet = info["dt_seconds"]
        dateo = info["dateo"]

        #read the coordinate values from
        if info["varname"].strip().lower() == var_name_with_target_coords.lower():
            npas_for_sl = npas
            deet_for_sl = deet
            dateo_for_sl = dateo
            ips_for_sl = ips
            igs_for_sl = info["ig"]
            nbits_for_sl = nbits
            data_type_for_sl = data_type
            typ_var_for_sl = info["var_type"]
            grid_type_for_sl = info["grid_type"]

        r_obj_out.write_2D_field(name=info["varname"],
                                 data=data, ip=ips,
                                 ig=info["ig"],
                                 npas=npas, deet=deet, label="", dateo=dateo,
                                 grid_type=info["grid_type"], typ_var=info["var_type"],
                                 nbits=nbits, data_type=data_type)
        i += 1


    #check that all fields were copied
    n_recs_in = r_obj_in.get_number_of_records()
    assert i == n_recs_in, "copied {0} records, but should be {1}".format(i, n_recs_in)

    #get coordinates
    r_obj_in.get_first_record_for_name(var_name_with_target_coords)
    lons2d_target, lats2d_target = r_obj_in.get_longitudes_and_latitudes_for_the_last_read_rec()
    #lons2d_target[lons2d_target >= 180] -= 360

    #Interpolate and save interflow slopes
    mat = None
    params, lons2d_source, lat2d_source = None, None, None
    ktree = None
    for sc, med in SLOPE_CLASS_TO_MEDIAN.iteritems():
        inpath = in_path_template.format(sc)
        if mat is None:
            params, lons2d_source, lat2d_source = _get_source_lon_lat(path=inpath)
            print lons2d_target.min(), lons2d_target.max()
            print lons2d_source.shape
            xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons2d_source.flatten(), lat2d_source.flatten())
            print "processing slopes of class sc={0}".format(sc)
            ktree = cKDTree(zip(xs, ys, zs))
            print "created kdtree.."

        tmp_mat = med * _get_slope_data(path=inpath)
        if mat is None:
            mat = tmp_mat
        else:
            mat += tmp_mat

    mat /= 100.0
    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons2d_target.flatten(), lats2d_target.flatten())

    #compare length scales
    a_source = distance.GreatCircleDistance(
        (lat2d_source[0, 0], lons2d_source[0, 0]),
        (lat2d_source[-1, -1], lons2d_source[-1, -1])).km ** 2 / np.prod(lons2d_source.shape)

    a_target = distance.GreatCircleDistance(
        (lats2d_target[0, 0], lons2d_target[0, 0]),
        (lats2d_target[-1, -1], lons2d_target[-1, -1])).km ** 2 / np.prod(lons2d_target.shape)

    nnegihbours = min(int(a_target / a_source), 2)
    print "nneighbours = {0}".format(nnegihbours)

    dists, inds = ktree.query(zip(xt, yt, zt), k=nnegihbours)
    interpolated_slopes = mat.toarray().flatten()[inds].mean(axis=0).reshape(lons2d_target.shape)

    r_obj_out.write_2D_field(name="ITFS",
                             data=interpolated_slopes, ip=ips_for_sl,
                             ig=igs_for_sl,
                             npas=npas_for_sl, deet=deet_for_sl, label="",
                             dateo=dateo_for_sl,
                             grid_type=grid_type_for_sl,
                             typ_var=typ_var_for_sl,
                             nbits=nbits_for_sl, data_type=data_type_for_sl)

    r_obj_in.close()
    r_obj_out.close()


def main(use_half_of_cols=True):
    folder_path = "/home/huziy/skynet3_rech1/Global_terrain_slopes_30s"
    out_filename = "slopes_30s.nc"
    in_fname_pattern = "GloSlopesCl{0}_30as.asc"

    nclasses = 1
    params = {}

    out_path = os.path.join(folder_path, out_filename)
    ds = Dataset(out_path, "w")

    for cl in range(1, nclasses + 1):
        print "cl = {0}".format(cl)
        inpath = os.path.join(folder_path, in_fname_pattern.format(cl))
        if cl == 1:  # generate lon/lat
            params, lon2d, lat2d = _get_source_lon_lat(path=inpath)
            nx, ny = lon2d.shape
            ds.createDimension("x", nx)
            ds.createDimension("y", ny)

            lon_var = ds.createVariable("lon", "f4", ("y", "x"))
            lat_var = ds.createVariable("lat", "f4", ("y", "x"))

            lon_var[:] = lon2d
            lat_var[:] = lat2d

        data_var = ds.createVariable("class{0}".format(cl), np.uint8, ("y", "x"), chunksizes=(10000, 10000))
        data_var.missing_value = params["NODATA_value"]
        data_var[:] = _get_slope_data(path=inpath)

    ds.close()


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    #interpolate_slopes()
    import time

    t0 = time.time()
    interpolate_slopes(in_path_template="/home/huziy/skynet3_rech1/Global_terrain_slopes_30s/GloSlopesCl{0}_30as.asc",
                       in_path_rpn_geophy="/gs/project/ugh-612-aa/huziy/geophysics_fields/geophys_Quebec_0.1deg_260x260_with_dd_v6")
    print "Execution time is {0} seconds.".format(time.time() - t0)
