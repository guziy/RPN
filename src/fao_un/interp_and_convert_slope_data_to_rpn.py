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
    params["NODATA_value"] = int(params["NODATA_value"])

    d = params["cellsize"]
    lon1d = [params["xllcorner"] + i * d for i in range(params["ncols"])]

    lat1d = [params["yllcorner"] + i * d for i in range(params["nrows"])]
    return params, np.asarray(lon1d), np.asarray(lat1d)


def _get_slope_data(path=""):
    """
    cols - the list of column indices to be read
    """
    data = np.loadtxt(path, skiprows=6, dtype=np.uint8, usecols=range(100))
    return np.flipud(data)[:100, :100]


def _get_slope_data_by_cols_and_rows(path="", cols=None, rows=None, missing_value = 255, 
                                     nrows_total = None, ncols_total = None):
    # with open(path) as f:
    #     lines = f.readlines()[6:][::-1]
    #     data = [[int(lines[row].split()[col]) for row in rows] for col in cols]

    """

    :param path:
    :return: numpy.ndarray
    """
    imin = min(rows)
    imax = max(rows)

    data = np.genfromtxt(path, skip_header=nrows_total - (imax+1) + 6, 
                         skip_footer = imin,
                         dtype=np.uint8, usecols=cols, missing_values = missing_value, 
                         usemask = True)

    if rows is not None:
        rows = np.asarray(rows)
        return np.flipud(data)

    return np.flipud(data)


def fill_missing_values(route_slope, interpolated_slopes,
                        lons2d = None, lats2d = None):


    to_fill = (route_slope >= 0) & (interpolated_slopes < 0)
    correct_slopes = interpolated_slopes >= 0

    x, y, z = lat_lon.lon_lat_to_cartesian(lons2d[correct_slopes], lats2d[correct_slopes])
    ktree = cKDTree(zip(x, y, z))



    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons2d[to_fill], lats2d[to_fill])
    dists, inds = ktree.query(zip(xt, yt, zt))

    interpolated_slopes[to_fill] = interpolated_slopes[correct_slopes][inds]



def interpolate_slopes(in_path_template="",
                       in_path_rpn_geophy="/skynet3_rech1/huziy/geof_lake_infl_exp/geophys_Quebec_0.1deg_260x260_with_dd_v6",
                       out_path_rpn_geophy=None, var_name_with_target_coords="SLOP", delta_deg=0.05):
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
    route_slope = None

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
            route_slope = data[:]

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
    lons2d_target[lons2d_target >= 180] -= 360

    #Interpolate and save interflow slopes
    mat = None
    params = None

    index_map = {}
    imin, jmin = np.Inf, np.Inf
    imax, jmax = -1, -1
    for sc, med in SLOPE_CLASS_TO_MEDIAN.iteritems():
        inpath = in_path_template.format(sc)
        if mat is None:
            params, lons1d_source, lats1d_source = _get_source_lon_lat(path=inpath)
            print params
            #build the map of closest indices
            nx, ny = lons2d_target.shape
            for i in range(nx):
                #if not (100 <= i <= 160): continue
                for j in range(ny):
                    #if not (100 <= j <= 160): continue
                    lon_target, lat_target = lons2d_target[i, j], lats2d_target[i, j]
                    dlon1 = np.abs(lons1d_source - lon_target)
                    dlon2 = np.abs(lons1d_source - 360 - lon_target) if lon_target < 0 else \
                        np.abs(lons1d_source + 360 - lon_target)

                    dlat = np.abs(lats1d_source - lat_target)
                    inds_i = np.where((dlon1 < delta_deg) |
                                      (dlon2 < delta_deg))[0]

                    inds_j = np.where(dlat < delta_deg)[0]
                    assert len(inds_i) > 0 and len(inds_j) > 0

                    imin, jmin = min(imin, inds_i.min()), min(jmin, inds_j.min())
                    imax, jmax = max(imax, inds_i.max()), max(jmax, inds_j.max())

                    inds_i, inds_j = np.meshgrid(inds_i, inds_j)

                    index_map[(i, j)] = [inds_j.flatten(), inds_i.flatten()]
                    #print i, j
        #i <-> longitude <-> columns in the file
        #j <-> latitude <-> rows in the file
        tmp_mat = _get_slope_data_by_cols_and_rows(path=inpath,
                                                   cols=range(imin, imax + 1),
                                                   rows=range(jmin, jmax + 1),
                                                   missing_value=params["NODATA_value"],
                                                   ncols_total = params["ncols"], nrows_total = params["nrows"])
        #tmp_mat = tmp_mat.astype(int)

        #nodata_pts = (tmp_mat == params["NODATA_value"])
        #tmp_mat = np.ma.masked_where(nodata_pts, tmp_mat)
        print tmp_mat.shape
        print tmp_mat.min(), tmp_mat.max(), tmp_mat.mean()
        if mat is None:
            mat = med * tmp_mat
        else:
            mat += med * tmp_mat

    mat /= 100.0


    #compare length scales
    interpolated_slopes = np.zeros_like(lons2d_target)

    for ij, inds in index_map.iteritems():
        data = mat[inds[0] - jmin, inds[1] - imin]
        print len(data[~data.mask]), np.prod(data.shape, dtype=np.float32), data.shape
        if float(len(data[~data.mask])) / np.prod(data.shape, dtype=np.float32) < 0.4:
            interpolated_slopes[ij[0], ij[1]] = -1
        else:
            interpolated_slopes[ij[0], ij[1]] = data[~data.mask].mean()

    print "ITFS: ", interpolated_slopes.min(), interpolated_slopes.max()



    fill_missing_values(route_slope, interpolated_slopes,
                        lons2d = lons2d_target, lats2d = lats2d_target)





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
                       in_path_rpn_geophy="/skynet3_rech1/huziy/geof_lake_infl_exp/geophys_Quebec_0.1deg_260x260_with_dd_v6")
    print "Execution time is {0} seconds.".format(time.time() - t0)
