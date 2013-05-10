from multiprocessing.pool import Pool
import os.path

__author__="huziy"
__date__ ="$Jul 25, 2011 4:56:03 PM$"


import netCDF4 as nc
import application_properties

from rpn.rpn import RPN

import os
import numpy as np


def extract_field(name = "VF", level = 3, in_file = "", out_file = None, margin = 0):
    if out_file is None:
        out_file = in_file + "_lf.nc"

    rObj = RPN(in_file)
    field = rObj.get_first_record_for_name_and_level(varname=name, level=level)
    lons2d, lats2d = rObj.get_longitudes_and_latitudes_for_the_last_read_rec()
    rObj.close()


    lons2d[lons2d > 180] -= 360.0


    ds = nc.Dataset(out_file, "w", format="NETCDF3_CLASSIC")

    nx, ny = field.shape

    ds.createDimension("lon", nx - margin)
    ds.createDimension("lat", ny - margin)

    var = ds.createVariable(name, "f4", dimensions=("lon", "lat"))
    lonVar = ds.createVariable("longitude", "f4", dimensions=("lon", "lat"))
    latVar = ds.createVariable("latitude", "f4", dimensions=("lon", "lat"))

    var[:] = field[:nx - margin, :ny - margin]
    var[:] = field[:nx - margin, :ny - margin]
    lonVar[:] = lons2d[:nx - margin, :ny - margin]
    latVar[:] = lats2d[:nx - margin,:ny - margin]
    ds.close()


    pass

def extract_runoff_to_netcdf_file(filePath = 'data/pm1957090100_00589248p', outDir = None):
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



def extract_runoff_to_nc_process(args):
    inPath, outPath = args

    #print "in: {0}".format( inPath )
    #print "out: {0}".format( outPath )

    traf_name = "TRAF"
    tdra_name = "TDRA"

    r = RPN(inPath)
    r.suppress_log_messages()
    traf_data = r.get_all_time_records_for_name(varname = traf_name)
    tdra_data = r.get_all_time_records_for_name(varname = tdra_name)
    r.close()


    nx, ny = traf_data.items()[0][1].shape


    ds = nc.Dataset(outPath, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("lon", nx)
    ds.createDimension("lat", ny)
    ds.createDimension("time", None)

    varTraf = ds.createVariable(traf_name, "f4", dimensions=("time", "lon", "lat"))
    varTraf.units = "kg/( m**2 * s )"

    varTdra = ds.createVariable(tdra_name, "f4", dimensions=("time", "lon", "lat"))
    varTdra.units = "kg/( m**2 * s )"


    timeVar = ds.createVariable("time", "f4", dimensions=("time",))


    sorted_dates = list( sorted(traf_data.keys()) )

    timeVar.units = "hours since {0}".format(sorted_dates[0])
    timeVar[:] = nc.date2num(sorted_dates, timeVar.units)


    varTraf[:] = np.array(
        [ traf_data[d] for d in sorted_dates ]
    )

    varTdra[:] = np.array(
        [ tdra_data[d] for d in sorted_dates ]
    )
    ds.close()




def runoff_to_netcdf_parallel(inDir, outDir):
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    inNames = [ x for x in os.listdir(inDir) if x.startswith("pm") and x.endswith("p") ]

    inPaths = [os.path.join(inDir, name) for name in inNames]

    print inPaths

    outPaths = [ os.path.join(outDir, inName + ".nc") for inName in inNames ]


    ppool = Pool(processes=10)
    ppool.map(extract_runoff_to_nc_process, zip(inPaths, outPaths) )





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

#
#    extract_field(name="VF", level=3, in_file="/home/huziy/skynet3_rech1/test/geophys_Quebec_86x86_0.5deg.v3")
#    extract_field(name="VF", level=3, in_file="/home/huziy/skynet3_rech1/test/geophys_Quebec_260x260_3")


    #extract_sand_and_clay_from_rpn(rpn_path= "/b2_fs2/huziy/OMSC26_MPI_long_new_v01/geo_Arctic_198x186",
    #    outpath="/home/huziy/skynet3_rech1/runoff_arctic_nc/geo_Arctic_198x186.nc")
    runoff_to_netcdf_parallel("/b2_fs2/huziy/OMSC26_Can_long_new_v01/", "/skynet3_rech1/huziy/runoff_arctic_nc/CanESM")

    print "Hello World"
