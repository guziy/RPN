import os.path

__author__="huziy"
__date__ ="$Jul 25, 2011 4:56:03 PM$"


import netCDF4 as nc
import application_properties

from rpn.rpn import RPN

import os
import numpy as np


def extract_field(name = "VF", level = 3, in_file = "", out_file = None):
    if out_file is None:
        out_file = in_file + "_lf.nc"

    rObj = RPN(in_file)
    field = rObj.get_first_record_for_name_and_level(varname=name, level=level)
    lons2d, lats2d = rObj.get_longitudes_and_latitudes_for_the_last_read_rec()
    rObj.close()


    lons2d[lons2d > 180] -= 360.0


    ds = nc.Dataset(out_file, "w", format="NETCDF3_CLASSIC")

    nx, ny = field.shape

    ds.createDimension("lon", nx - 2)
    ds.createDimension("lat", ny - 2)

    var = ds.createVariable(name, "f4", dimensions=("lon", "lat"))
    lonVar = ds.createVariable("longitude", "f4", dimensions=("lon", "lat"))
    latVar = ds.createVariable("latitude", "f4", dimensions=("lon", "lat"))

    var[:] = field[:-2, :-2]
    lonVar[:] = lons2d[:-2, :-2]
    latVar[:] = lats2d[:-2,:-2]
    ds.close()


    pass

def extract_runoff_to_netcdf_file(filePath = 'data/pm1957090100_00589248p'):
    surface_runoff_name = 'TRAF'
    subsurface_runoff_name = 'TDRA'
    level_tdra = 5
    level_traf = 5


    print filePath

    #get data from the rpn file
    rpnObj = RPN(filePath)

    assert rpnObj.get_number_of_records() > 4, filePath
    surfRunoff = rpnObj.get_first_record_for_name_and_level(surface_runoff_name, level = level_traf)
    subSurfRunoff = rpnObj.get_first_record_for_name_and_level(subsurface_runoff_name, level = level_tdra)

    nx, ny = surfRunoff.shape
    ncFile = nc.Dataset(filePath + '.nc', 'w', format = 'NETCDF3_CLASSIC')
    ncFile.createDimension('lon', nx)
    ncFile.createDimension('lat', ny)

    surfRunoffVar = ncFile.createVariable(surface_runoff_name, 'f', ('lon','lat'))
    subSurfRunoffVar = ncFile.createVariable(subsurface_runoff_name, 'f', ('lon','lat'))

    subSurfRunoffVar[:] = subSurfRunoff
    surfRunoffVar[:] = surfRunoff
    ncFile.forecast_hour = rpnObj.get_current_validity_date()
    ncFile.close()

    rpnObj.close()
    #os.remove(filePath)


def extract_runoff_to_netcdf_folder(folder_path = 'data/CORDEX/Africa/Samples'):
    for folder in os.listdir(folder_path):
        monthFolderPath = os.path.join(folder_path, folder)

        if not os.path.isdir(monthFolderPath):
            continue
        for fileName in os.listdir(monthFolderPath):

            if fileName.endswith('.nc'):
                continue
            filePath = os.path.join(monthFolderPath, fileName)
            extract_runoff_to_netcdf_file(filePath)


def extract_sand_and_clay_from_rpn(rpn_path = 'data/geophys_africa', outpath = ""):
    rpnFile = RPN(rpn_path)
    sandField = rpnFile.get_2D_field_on_all_levels('SAND')
    clayField = rpnFile.get_2D_field_on_all_levels('CLAY')
    dpthField = rpnFile.get_first_record_for_name("8L")
    rpnFile.close()

    ncFile = nc.Dataset(outpath, 'w', format = 'NETCDF3_CLASSIC')

    
    nx, ny = sandField[1].shape
    nz = len(sandField)

    print nx, ny, nz
    print sandField[1].shape

    sand = np.zeros((nx, ny, nz))
    clay = np.zeros((nx, ny, nz))


    for i in xrange(nz):
        print i
        sand[:, :, i] = sandField[ i + 1 ][:,:]
        clay[:, :, i] = clayField[ i + 1 ][:,:]

        print 'SAND:',i, np.min(sand[:, :, i]), np.max(sand[:, :, i])
        print 'CLAY:',i, np.min(clay[:, :, i]), np.max(clay[:, :, i])

    ncFile.createDimension('lon', nx)
    ncFile.createDimension('lat', ny)
    ncFile.createDimension('level', nz)

    sandVar = ncFile.createVariable('SAND', 'f', ('lon','lat','level'))
    clayVar = ncFile.createVariable('CLAY', 'f', ('lon','lat','level'))
    dpthVar = ncFile.createVariable('DPTH_TO_BEDROCK', 'f', ('lon','lat'))

    sandVar[:] = sand
    clayVar[:] = clay
    dpthVar[:] = dpthField
    ncFile.close()


    pass

def delete_files_with_nrecords(folder_path = 'data/CORDEX/Africa/Samples', n_records = 4):
    for folder in os.listdir(folder_path):
        monthFolderPath = os.path.join(folder_path, folder)
        if not os.path.isdir(monthFolderPath):
            continue

        for fileName in os.listdir(monthFolderPath):
            filePath = os.path.join(monthFolderPath, fileName)
            rpnObj = RPN(filePath)
            delete = False
            if rpnObj.get_number_of_records()  == n_records:
                delete = True
            rpnObj.close()
            if delete:
                os.remove(filePath)
                print 'removing %s' % filePath

    pass



if __name__ == "__main__":
    application_properties.set_current_directory()
#    extract_sand_and_clay_from_rpn(rpn_path = 'data/CORDEX/NA/NA_CLASS_L03_v3321_195709/pm1957090100_00000000p')
#    extract_sand_and_clay_from_rpn(rpn_path="/home/huziy/skynet3_exec1/from_guillimin/Africa_044deg_geophy/pm1975120100_00000000p",
#        outpath="africa_0.44deg_sand_clay_dpth.nc"
#    )
#    delete_files_with_nrecords()
#    extract_runoff_to_netcdf_folder(folder_path = 'data/CORDEX/NA_fix')


    extract_field(name="VF", level=3, in_file="/home/huziy/skynet3_rech1/test/geophys_Quebec_86x86_0.5deg.v3")

    print "Hello World"
