__author__ = 'huziy'
import gdal
from gdalconst import *
import numpy as np

from netCDF4 import Dataset


def main():
    path = "/skynet1_rech3/huziy/arlette_test/gl_pasture_grid/gl_pasture/w001001.adf"

    gd = gdal.Open(path, GA_ReadOnly)
    assert isinstance(gd, gdal.Dataset)
    data = gd.ReadAsArray()
    print np.min(data), np.max(data)
    print data.shape
    print type(data)

    print gd.RasterXSize, gd.RasterYSize

    geo_transform = gd.GetGeoTransform()

    lon_1d = np.arange(geo_transform[0], geo_transform[0] + geo_transform[1] * gd.RasterXSize,
                       geo_transform[1])
    lat_1d = np.arange(geo_transform[3], geo_transform[3] + geo_transform[5] * gd.RasterYSize,
                       geo_transform[5])

    lat_1d = lat_1d[::-1]

    print lon_1d.shape, lat_1d.shape


    data = np.flipud(data)

    data = np.ma.masked_where((data < 0) | (data > 1), data)
    data[data.mask] = data.fill_value

    out_path = "/skynet1_rech3/huziy/arlette_test/gl_pasture.nc"
    ds = Dataset(out_path, "w", format="NETCDF3_CLASSIC")

    ds.createDimension("lon", size=gd.RasterXSize)
    ds.createDimension("lat", size=gd.RasterYSize)
    pasture = ds.createVariable("pasture", "f4", dimensions=("lat", "lon"))
    pasture[:] = data
    pasture.missing_value = data.get_fill_value()


    lon2d_var = ds.createVariable("longitude", "f4", dimensions=("lat", "lon"))
    lat2d_var = ds.createVariable("latitude", "f4", dimensions=("lat", "lon"))

    lon2d_var[:], lat2d_var[:] = np.meshgrid(lon_1d, lat_1d)
    ds.close()


if __name__ == '__main__':
    main()
