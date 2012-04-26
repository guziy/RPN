from multiprocessing import Pool
from netCDF4 import Dataset
import os
import application_properties
import draw_regions
from active_layer_thickness import CRCMDataManager

__author__ = 'huziy'

import numpy as np


def _get_average(x):
    """
    for use in process pool
    """
    dm, year, var_name, months = x
    return dm.get_seasonal_mean_for_year_of_2d_var(year, months=months, var_name=var_name)

def main(nc_path = "swe_era40-interim.nc"):
    var_name = "I5"
    nc_var_name = "swe"
    #data_folder = "/skynet1_rech3/huziy/cordex/CRCM5_output/North_America/NorthAmerica_0.44deg_CanRCP45E1"
    data_folder = "/home/huziy/skynet1_rech3/cordex/for_Sheena/era40_era-interim_driven"
    dm = CRCMDataManager(data_folder=data_folder)

    year_range = range(1958, 2009)

    pool = Pool(processes=10)
    n_years = len(year_range)
    months = [1,2,12]
    input = zip([dm] * n_years, year_range, [var_name] * n_years, [months] * n_years)
    data = pool.map(_get_average, input)
    data = np.array(data)

    #coord_file = "/skynet1_rech3/huziy/cordex/CRCM5_output/North_America/NorthAmerica_0.44deg_CanRCP45E1/Samples/NorthAmerica_0.44deg_CanRCP45E1_198001/"
    coord_file = os.path.join(data_folder, "pmNorthAmerica_0.44deg_ERA40-Int2_195801_moyenne")
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(file_path=coord_file)


    ds = Dataset(nc_path, mode = "w", format="NETCDF3_CLASSIC")
    ds.createDimension('year', len(year_range))
    ds.createDimension('lon', lons2d.shape[0])
    ds.createDimension('lat', lons2d.shape[1])

    lonVariable = ds.createVariable('longitude', 'f4', ('lon', 'lat'))
    latVariable = ds.createVariable('latitude', 'f4', ('lon', 'lat'))
    yearVariable = ds.createVariable("year", "i4", ("year",))

    theVariable = ds.createVariable(nc_var_name, "f4", ('year','lon', 'lat'))

    lonVariable[:,:] = lons2d[:,:]
    latVariable[:,:] = lats2d[:,:]
    yearVariable[:] = year_range

    theVariable[:,:,:] = data[:,:,:]
    theVariable.units = "mm of water"
    ds.close()

    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  