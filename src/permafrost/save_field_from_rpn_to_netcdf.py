from multiprocessing import Pool
from netCDF4 import Dataset
import os
from scipy.spatial.kdtree import KDTree
import application_properties
import draw_regions
from active_layer_thickness import CRCMDataManager
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np


def _get_average(x):
    """
    for use in process pool
    """
    dm, year, var_name, months = x
    return dm.get_seasonal_mean_for_year_of_2d_var(year, months=months, var_name=var_name)

def main(nc_path = "swe_era40_b1.nc"):
    var_name = "I5"
    nc_var_name = "swe"
    #data_folder = "/skynet1_rech3/huziy/cordex/CRCM5_output/North_America/NorthAmerica_0.44deg_CanRCP45E1"
    #data_folder = "/home/huziy/skynet1_rech3/cordex/for_Sheena/era40_era-interim_driven"
    #data_folder = "/skynet1_rech3/huziy/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_MPI_B1_dm"
    data_folder = "/skynet1_rech3/huziy/cordex/CORDEX_DIAG/era40_driven_b1"

    year_range = range(1958, 2009)


    dm = CRCMDataManager(data_folder=data_folder)
    pool = Pool(processes=10)
    n_years = len(year_range)
    months = [12,1,2]
    input = zip([dm] * n_years, year_range, [var_name] * n_years, [months] * n_years)
    data = pool.map(_get_average, input)
    data = np.array(data)

    coord_file = "/skynet1_rech3/huziy/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1"
    coord_file = os.path.join(coord_file, "pmNorthAmerica_0.44deg_CanHisto_B1_195801_moyenne")
    #coord_file = os.path.join(data_folder, "pmNorthAmerica_0.44deg_ERA40-Int2_195801_moyenne")
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

def test_evol_for_point():
    lon = -100
    lat = 60

    path = "/home/huziy/skynet1_rech3/cordex/for_Samira/b1/tmp_era40_b1.nc"

    ds = Dataset(path)
    data = ds.variables["tmp"][:]
    years = ds.variables["year"][:]

    coord_file = "/skynet1_rech3/huziy/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1"
    coord_file = os.path.join(coord_file, "pmNorthAmerica_0.44deg_CanHisto_B1_195801_moyenne")
    #coord_file = os.path.join(data_folder, "pmNorthAmerica_0.44deg_ERA40-Int2_195801_moyenne")
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(file_path=coord_file)

    sel_lons = [lon]
    sel_lats = [lat]
    xo,yo,zo = lat_lon.lon_lat_to_cartesian(sel_lons, sel_lats)

    xi, yi, zi = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
    ktree = KDTree(zip(xi,yi,zi))
    dists, indexes =  ktree.query(zip(xo,yo,zo))


    print len(indexes)
    print indexes
    idx = indexes[0]

    import matplotlib.pyplot as plt
    plt.figure()
    data_to_show = []

    for i, y in enumerate(years):

        data_to_show.append(data[i,:,:].flatten()[idx])

    plt.plot(years, data_to_show, "-s", lw = 3)
    plt.grid()

    plt.show()



    pass
if __name__ == "__main__":
    application_properties.set_current_directory()
    test_evol_for_point()
    #main()
    print "Hello world"
  