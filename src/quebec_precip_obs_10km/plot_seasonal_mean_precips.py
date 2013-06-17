import calendar
from datetime import datetime
from multiprocessing.pool import Pool
import os
from crcm5.model_data import Crcm5ModelDataManager

__author__ = 'huziy'

import numpy as np
DATA_FOLDER = "/home/huziy/skynet3_rech1/kvl_bis"
FILE_NAME_FORMAT = "grid_prcp_{0}-{1}-{2}-kvl.txt"

from osgeo import osr, ogr

def get_lat_lons():
    fName = os.listdir(DATA_FOLDER)[0]
    fPath = os.path.join(DATA_FOLDER, fName)

    f = open(fPath)

    descr = {}
    line = f.readline()
    while line != "":
        line = line.strip()
        if line == "": continue


        fields = [ x.strip() for x in line.split() ]

        if fields[0] in ["ncols", "nrows"]:
            descr[fields[0]] = int(fields[1])


        if fields[0] in ["xllcenter", "yllcenter", "cellsize", "nodata_value"]:
            descr[fields[0]] = float(fields[1])
        print descr
        if len(descr) == 6: break
        line = f.readline()

    print "Read header"
    f.close()




    utm_srs = osr.SpatialReference()

    utm_srs.SetUTM( 18 )

    utm_srs.SetWellKnownGeogCS( 'WGS84' )
    ll_srs = osr.SpatialReference()
    ll_srs.SetWellKnownGeogCS( 'WGS84' )

    ct = osr.CoordinateTransformation( utm_srs, ll_srs )

    x = [descr["xllcenter"] + i * descr["cellsize"] for i in range(descr["ncols"])]
    y = [descr["yllcenter"] + i * descr["cellsize"] for i in range(descr["nrows"])]

    y2d, x2d = np.meshgrid(y, x)

    x_flat, y_flat = x2d.flatten(), y2d.flatten()


    print "creating points"
    points = [
        ogr.CreateGeometryFromWkt( "POINT({0} {1})".format(the_x, the_y), utm_srs )
            for the_x, the_y in zip(x_flat, y_flat)
    ]


    statuses = [ p.Transform(ct) for p in points ]
    lons = np.array( [ p.GetX() for p in points ] )
    lats = np.array( [ p.GetY() for p in points ] )

    lons2d = lons.reshape(x2d.shape)
    lats2d = lats.reshape(y2d.shape)

    return lons2d, lats2d, descr



def calculate_seasonal_mean_field(months = None, start_year = None, end_year = None):
    fPathFormat = os.path.join(DATA_FOLDER, FILE_NAME_FORMAT)
    all_data = []
    for year in range(start_year, end_year + 1):
        for month in months:
            for d in range(1, calendar.monthrange(year, month)[1] + 1 ):
                fPath = fPathFormat.format(year, month, d)
                data = np.flipud( np.loadtxt(fPath, skiprows=6) ).transpose()
                all_data.append(data)

    return np.mean(all_data, axis = 0)
    pass


def main():
    import matplotlib.pyplot as plt
    lons, lats, descr = get_lat_lons()

    #needed for the model domain
    manager = Crcm5ModelDataManager(samples_folder_path="/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup",
        all_files_in_samples_folder = True
    )

    b = manager.get_rotpole_basemap()
    x, y = b(lons, lats)

    start_year = 1979
    end_year = 1988

    data = calculate_seasonal_mean_field(months=[6,7,8], start_year= start_year, end_year=end_year)

    levels = np.arange(0,7,0.25)
    data = np.ma.masked_where(data == descr["nodata_value"], data)

    b.contourf(x, y, data, levels = levels)
    b.colorbar()
    b.drawcoastlines()
    plt.show()

    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()

    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=20, height_cm=20, font_size=26)

    main()
    print "Hello world"
  