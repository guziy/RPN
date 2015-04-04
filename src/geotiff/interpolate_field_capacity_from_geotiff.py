from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN

__author__ = 'huziy'

from osgeo import gdal, gdalconst
import matplotlib.pyplot as plt
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.image import ImageContainerQuick, ImageContainerNearest

import crcm5.analyse_hdf.do_analysis_using_pytables as hdf_reader
import numpy as np


def read_observed_bfc(path="/home/huziy/skynet3_rech1/bulk_field_capacity_obs/wcs_ORNL_DAAC"):
    ds = gdal.Open(path, gdalconst.GA_ReadOnly)

    if ds is None:
        print("Could not open {}".format(path))


    assert isinstance(ds, gdal.Dataset)

    print(ds.GetDescription())
    extent = ds.GetGeoTransform()
    print(ds.GetGCPProjection())

    nx, ny = ds.RasterXSize, ds.RasterYSize

    dx, dy = extent[1], extent[5]
    print(extent)

    xll, yur = extent[0], extent[3]
    xur = xll + nx * dx
    yll = yur + ny * dy

    data = ds.ReadAsArray()

    # plt.pcolormesh(np.flipud(data))
    # plt.show()

    print(data.shape)
    print(nx, ny)

    print([xll, yll, xur, yur])

    adf = AreaDefinition("source", "lonlat", "lonlat", {"proj": "lonlat"}, nx, ny, [xll, yll, xur, yur])

    img = ImageContainerNearest(data, adf, 1000000)

    return img


def _get_depth_to_bedrock(path):
    r = RPN(path)
    data = r.get_first_record_for_name("8L")
    r.close()
    return data


def main(save_to_nc=True):
    """
    Read geotiff file with the field capacity field in mm,
    interpolate it to model grid, divide by the depth to bedrock,
    and save to netcdf file
    """

    # Read model data
    obs_fields_path = "/skynet3_rech1/huziy/geofields_interflow_exp/pm1979010100_00000000p"

    robj = RPN(obs_fields_path)

    thfc = robj.get_first_record_for_name_and_level(varname="D9", level=1)

    bdrck_depth = _get_depth_to_bedrock(obs_fields_path)

    proj_params = robj.get_proj_parameters_for_the_last_read_rec()

    rll = RotatedLatLon(**proj_params)

    t_lons, t_lats = robj.get_longitudes_and_latitudes_for_the_last_read_rec()

    t_lons[t_lons > 180] -= 360.0
    bmp = rll.get_basemap_object_for_lons_lats(lons2d=t_lons, lats2d=t_lats)

    robj.close()



    intp_img = read_observed_bfc().resample(
        SwathDefinition(t_lons.flatten(), t_lats.flatten())
    )
    fig = plt.figure(figsize=(10, 6))

    assert isinstance(fig, Figure)


    gs = GridSpec(1, 3, width_ratios=[0.95, 1, 1])


    # Observed bulk field capacity
    ax = fig.add_subplot(gs[0, 0])
    x, y = bmp(t_lons, t_lats)
    to_plot = intp_img.image_data.reshape(t_lons.shape)
    to_plot = np.ma.masked_where(to_plot < 0, to_plot) / (1.0e3 * bdrck_depth)

    im = bmp.pcolormesh(x, y, to_plot, vmin=0, vmax=0.5)
    bmp.drawcoastlines()
    ax.set_title("ORNL DAAC")

    # Currently used in the model
    ax = fig.add_subplot(gs[0, 1])
    thfc_to_plot = np.ma.masked_where(to_plot.mask, thfc)
    im = bmp.pcolormesh(x, y, thfc_to_plot, vmin=0, vmax=0.5)
    bmp.colorbar(im, ax=ax)
    bmp.drawcoastlines(ax=ax)
    ax.set_title("Current")
    bmp.colorbar(im, ax=ax)


    # Current - (ORNL DAAC)
    ax = fig.add_subplot(gs[0, 2])
    x, y = bmp(t_lons, t_lats)
    cmap = cm.get_cmap("RdBu_r", 21)
    im = bmp.pcolormesh(x, y, thfc_to_plot - to_plot, vmin=-0.5, vmax=0.5, cmap=cmap)
    bmp.colorbar(im, ax=ax)
    bmp.drawcoastlines(ax=ax)
    ax.set_title("Current - (ORNL-DAAC)")
    bmp.colorbar(im, ax=ax)



    plt.show()

    pass

if __name__ == '__main__':
    main()