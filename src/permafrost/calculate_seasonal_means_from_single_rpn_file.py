from netCDF4 import Dataset
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np





def main():
    #path = "/home/huziy/skynet3_rech1/test/snw_LImon_NA_CRCM5_CanESM2_historical_r1i1p1_185001-200512.rpn"
    path = "/home/sheena/skynet3_exec2/RPN/src/permafrost/snw_NA_CRCM5_CanESM2_rcp45_r1i1p1_200601-210012.rpn"
    months = [1,2,12]

    varname = "I5"

    rObj = RPN( path )
    records = rObj.get_all_time_records_for_name(varname=varname)
    lons2d, lats2d = rObj.get_longitudes_and_latitudes()

    rObj.close()



    times = sorted(records.keys())
    vals =  np.array( [records[t] for t in times])

    year_range = list(range(2006, 2101))
    nc_file_name = "{0:s}_{1:d}_{2:d}.nc".format(varname, year_range[0], year_range[-1])
    nx, ny = vals[0].shape


    #create netcdf file
    ds = Dataset(nc_file_name, "w", format = 'NETCDF3_CLASSIC')
    ds.createDimension('lon', nx)
    ds.createDimension('lat', ny)
    ds.createDimension("year", len(year_range))
    the_var = ds.createVariable(varname, 'f', ("year",'lat','lon'))
    the_lon = ds.createVariable("xlon", 'f', ('lat','lon'))
    the_lat = ds.createVariable("xlat", 'f', ('lat','lon'))



    for i, the_year in enumerate(year_range):
        bool_vector = [t.year == the_year and t.month in months for t in times]
        bool_vector = np.array(bool_vector)
        the_var[i,:,:] = np.mean(vals[bool_vector], axis=0).transpose()

    the_lon[:] = lons2d[:,:].transpose()
    the_lat[:] = lats2d[:,:].transpose()

    ds.close()










    #TODO: implement
    pass

import application_properties
if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print("Hello world")
  