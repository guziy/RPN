from datetime import datetime
import time
from osgeo.gdal import Dataset
from scipy.spatial.kdtree import KDTree
from domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
from multiprocessing import Pool

__author__ = 'huziy'

import numpy as np

import os
from osgeo import gdal
import matplotlib.pyplot as plt
from util.geo import lat_lon

def _get_outfilename(inPath):
    outPath = os.path.basename(inPath)
    fields = outPath.split(".")
    outPath = ".".join(fields[:-1]) + ".rpn"
    return outPath


name_to_nodata_value = {
    "gpp": 65535, "npp": 32767 , "qc": 255
}

name_to_mult = {
    "gpp": 0.1, "npp": 0.1 , "qc": 1.0
}

LIMIT_DIST = 50e3 #in meters
AGGR_SIZE = 2500 #number of the closest high res cells taken for aggragation to a single value of the low resolution grid

def convert(inPath, lonlats):

    ds = gdal.Open(inPath, gdal.GA_ReadOnly)
    assert isinstance(ds, Dataset)
    (Xul, deltaX, rotation, Yul, rotation, deltaY) = ds.GetGeoTransform()
    print(dir(ds))
    print(ds.GetMetadata_Dict())
    print(ds.GetDescription())

    srs_wkt = ds.GetProjection()
    Nx = ds.RasterXSize
    Ny = ds.RasterYSize
    print(ds.RasterCount)


    nxToRead = Nx / 2
    nyToRead = int(Ny / 1.5)

    data = ds.GetRasterBand(1).ReadAsArray(0, 0, nxToRead, nyToRead).transpose()
    print(srs_wkt)
    print(data.shape)

    #plt.imshow(data)
    #plt.show()
    ds = None

    print(Xul, Yul, deltaX, deltaY, rotation)


    x1d = np.arange(Xul, Xul + deltaX * nxToRead, deltaX)
    y1d = np.arange(Yul, Yul + deltaY * nyToRead, deltaY)

    assert len(x1d) == nxToRead
    assert len(y1d) == nyToRead



    y, x = np.meshgrid(y1d, x1d)




    fieldName = os.path.basename(inPath).split("_")[0].lower()


    coef = name_to_mult[fieldName]
    no_data = name_to_nodata_value[fieldName]
    usable = (data != no_data)

    print(x.shape, usable.shape)

    x0 = x[usable]
    y0 = y[usable]

    cartx, carty, cartz = lat_lon.lon_lat_to_cartesian(x0, y0)

    data_1d = data[usable]
    print("useful data points : {0}".format(len(x0)))

    tree = KDTree(list(zip(cartx, carty, cartz)))

    print("constructed the kdtree")

    xi, yi, zi = lat_lon.lon_lat_to_cartesian(lonlats[:,0], lonlats[:,1])
    dists, inds = tree.query(list(zip(xi, yi, zi)), k = AGGR_SIZE)


    npoints = dists.shape[0]
    interp_data = np.zeros((npoints, ))
    for i in range(npoints):
        the_dists = dists[i,:]
        the_inds = inds[i,:]

        good_pts = (the_dists < LIMIT_DIST)
        if len(the_dists[good_pts]) < 0.25 * AGGR_SIZE: #if there is no usable points in the vicinity, then set the value to no_data
            interp_data[i] = -1
            continue

        the_dists = the_dists[good_pts]
        the_inds = the_inds[good_pts]

        interp_coefs = 1.0 / the_dists ** 2
        interp_data[i] = np.sum( interp_coefs * data_1d[the_inds] ) / np.sum(interp_coefs)


    interp_data[interp_data >= 0] *= coef




    print("completed interpolation")
    return interp_data


def do_conversion_in_parallel(nprocs = 1):
    tiff_folder = "/home/huziy/skynet3_exec1/tiffs/"
    rpn_folder = "/home/huziy/skynet3_exec1/rpns_from_tifs"

    in_paths = []
    out_paths = []
    for tif_name in os.listdir(tiff_folder):

        if not tif_name.lower().endswith(".tif"): continue

        tif_path = os.path.join(tiff_folder, tif_name)
        rpn_path = os.path.join(rpn_folder, _get_outfilename(tif_path))
        in_paths.append(tif_path)
        out_paths.append(rpn_path)

    pool = Pool(processes=nprocs)
    pool.map(main, list(zip(in_paths, out_paths)))



    pass



def main(inout_paths):
    tiff_path, rpn_path = inout_paths
    print("tif path = {0}".format(tiff_path))
    print("rpn path = {0}".format(rpn_path))

    outGrid = RotatedLatLon(lon1=-90.0, lat1=50.0, lon2=0.0, lat2=0.0)
    Grd_dx  = 0.5
    Grd_dy  = 0.5
    Grd_ni  = 170
    Grd_nj  = 158
    Grd_iref = 11
    Grd_jref = 11
    Grd_latr = -33.5
    Grd_lonr = 140.5


    lons1d = np.array([Grd_lonr + (i - Grd_iref + 1) * Grd_dx for i in range(Grd_ni)])
    lats1d = np.array([Grd_latr + (j - Grd_jref + 1) * Grd_dy for j in range(Grd_nj)])


    lats2d, lons2d = np.meshgrid(lats1d, lons1d)

    lonlats = np.array( list(map( lambda x, y: outGrid.toGeographicLonLat(x, y), lons2d.flatten(), lats2d.flatten() )) )
    print(lonlats.shape)


    rObj = RPN(rpn_path, mode = "w")
    data = convert(tiff_path, lonlats)
    print("interpolated data")
    data.shape = lons2d.shape

    fieldName = os.path.basename(tiff_path).split("_")[0].lower()

    #write coordinates
    ig = outGrid.write_coords_to_rpn(rObj, lons1d, lats1d)

    rObj.write_2D_field(name = fieldName, data=data, grid_type="Z", ig = ig, label = fieldName)
    rObj.close()
    return 0

    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    t0 = time.clock()
    print("start time ", datetime.now())
    #main()

    do_conversion_in_parallel()
    print("end time ", datetime.now())
    t1 = time.clock()
    print("execution {0}".format(t1-t0))
    print("Hello world")
  
