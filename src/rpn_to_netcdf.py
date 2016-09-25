from multiprocessing.pool import Pool
import os.path

from pathlib import Path

from rpn import level_kinds
from rpn.rpn_multi import MultiRPN

from crcm5.mh_domains import default_domains
from crcm5.model_data import Crcm5ModelDataManager

__author__ = "huziy"
__date__ = "$Jul 25, 2011 4:56:03 PM$"

import netCDF4 as nc
import application_properties

from rpn.rpn import RPN

import os
import numpy as np


EXTEND_MARGIN = 20  # The number of gridpoints added from each side
TARGET_GRID_COVERAGE = default_domains.bc_mh_011


def extract_field(name="VF", level=3, in_file="", out_file=None, margin=0):
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
    latVar[:] = lats2d[:nx - margin, :ny - margin]
    ds.close()

    pass


def extract_runoff_to_netcdf_file(filePath='data/pm1957090100_00589248p', outDir=None):
    surface_runoff_name = 'TRAF'
    subsurface_runoff_name = 'TDRA'
    level_tdra = 5
    level_traf = 5

    print(filePath)

    #get data from the rpn file
    rpnObj = RPN(filePath)

    assert rpnObj.get_number_of_records() > 4, filePath
    surfRunoff = rpnObj.get_first_record_for_name_and_level(surface_runoff_name, level=level_traf)
    subSurfRunoff = rpnObj.get_first_record_for_name_and_level(subsurface_runoff_name, level=level_tdra)

    nx, ny = surfRunoff.shape

    ncFile = nc.Dataset(filePath + '.nc', 'w', format='NETCDF3_CLASSIC')
    ncFile.createDimension('lon', nx)
    ncFile.createDimension('lat', ny)

    surfRunoffVar = ncFile.createVariable(surface_runoff_name, 'f', ('lon', 'lat'))
    subSurfRunoffVar = ncFile.createVariable(subsurface_runoff_name, 'f', ('lon', 'lat'))

    subSurfRunoffVar[:] = subSurfRunoff
    surfRunoffVar[:] = surfRunoff
    ncFile.forecast_hour = rpnObj.get_current_validity_date()
    ncFile.close()

    rpnObj.close()
    #os.remove(filePath)


def extract_runoff_to_netcdf_folder(folder_path='data/CORDEX/Africa/Samples'):
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
    in_path, out_path = args

    if os.path.exists(out_path):
        print("Nothing to do for: {}".format(out_path))
        return  # skip files that already exist


    traf_name = "TRAF"
    tdra_name = "TDRA"

    r = MultiRPN(in_path)
    traf_data = r.get_all_time_records_for_name_and_level(varname=traf_name, level=5, level_kind=level_kinds.ARBITRARY)
    tdra_data = r.get_all_time_records_for_name_and_level(varname=tdra_name, level=5, level_kind=level_kinds.ARBITRARY)
    r.close()

    nx, ny = list(traf_data.items())[0][1].shape

    with nc.Dataset(out_path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.createDimension("lon", nx)
        ds.createDimension("lat", ny)
        ds.createDimension("time", None)

        varTraf = ds.createVariable(traf_name, "f4", dimensions=("time", "lon", "lat"))
        varTraf.units = "kg/( m**2 * s )"

        varTdra = ds.createVariable(tdra_name, "f4", dimensions=("time", "lon", "lat"))
        varTdra.units = "kg/( m**2 * s )"

        timeVar = ds.createVariable("time", "f4", dimensions=("time",))

        sorted_dates = list(sorted(traf_data.keys()))

        timeVar.units = "hours since {0}".format(sorted_dates[0])
        timeVar[:] = nc.date2num(sorted_dates, timeVar.units)

        varTraf[:] = np.array(
            [traf_data[d] for d in sorted_dates]
        )

        varTdra[:] = np.array(
            [tdra_data[d] for d in sorted_dates]
        )


def runoff_to_netcdf_parallel(indir, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    in_names = [x for x in os.listdir(indir) if x.startswith("pm") and x.endswith("p")]

    in_paths = [os.path.join(indir, name) for name in in_names]

    out_paths = [os.path.join(outdir, inName + ".nc") for inName in in_names]

    ppool = Pool(processes=10)
    print("The paths below go to: ")
    print(in_paths[0])
    print("Go into: {}".format(out_paths[0]))
    ppool.map(extract_runoff_to_nc_process, list(zip(in_paths, out_paths)))


def runoff_to_netcdf_parallel_with_multirpn(indir, outdir):

    indir_p = Path(indir)
    outdir_p = Path(outdir)


    if not os.path.isdir(outdir):
        os.mkdir(outdir)



    in_paths = [[str(x) for x in monthdir.iterdir() if x.name.startswith("pm") and x.name.endswith("p")] for monthdir in indir_p.iterdir() if monthdir.is_dir()]


    out_paths = [str(outdir_p.joinpath(monthdir.name + ".nc")) for monthdir in indir_p.iterdir() if monthdir.is_dir()]

    ppool = Pool(processes=10)
    ppool.map(extract_runoff_to_nc_process, list(zip(in_paths, out_paths)))


def extract_sand_and_clay_from_rpn(rpn_path='data/geophys_africa', outpath=""):
    rpn_file = RPN(rpn_path)
    sandField = rpn_file.get_2D_field_on_all_levels('SAND')
    clayField = rpn_file.get_2D_field_on_all_levels('CLAY')
    dpthField = rpn_file.get_first_record_for_name("8L")
    rpn_file.close()

    nc_file = nc.Dataset(outpath, 'w', format='NETCDF3_CLASSIC')

    nx, ny = sandField[1].shape
    nz = len(sandField)

    sand = np.zeros((nx, ny, nz))
    clay = np.zeros((nx, ny, nz))

    for i in range(nz):
        print(i)
        sand[:, :, i] = sandField[i + 1][:, :]
        clay[:, :, i] = clayField[i + 1][:, :]


    lon_dim_name = "longitude"
    lat_dim_name = "latitude"

    nc_file.createDimension(lon_dim_name, nx)
    nc_file.createDimension(lat_dim_name, ny)
    nc_file.createDimension('level', nz)

    sand_var = nc_file.createVariable("SAND", "f4", (lon_dim_name, lat_dim_name, "level"))
    clay_var = nc_file.createVariable("CLAY", "f4", (lon_dim_name, lat_dim_name, "level"))
    dpth_var = nc_file.createVariable("DEPTH_TO_BEDROCK", "f4", (lon_dim_name, lat_dim_name))

    sand_var[:] = sand
    clay_var[:] = clay
    dpth_var[:] = dpthField
    nc_file.close()


def delete_files_with_nrecords(folder_path='data/CORDEX/Africa/Samples', n_records=4):
    for folder in os.listdir(folder_path):
        monthFolderPath = os.path.join(folder_path, folder)
        if not os.path.isdir(monthFolderPath):
            continue

        for fileName in os.listdir(monthFolderPath):
            filePath = os.path.join(monthFolderPath, fileName)
            rpnObj = RPN(filePath)
            delete = False
            if rpnObj.get_number_of_records() == n_records:
                delete = True
            rpnObj.close()
            if delete:
                os.remove(filePath)
                print('removing %s' % filePath)

    pass


if __name__ == "__main__":
    application_properties.set_current_directory()
    #    extract_sand_and_clay_from_rpn(rpn_path = 'data/CORDEX/NA/NA_CLASS_L03_v3321_195709/pm1957090100_00000000p')
    #extract_sand_and_clay_from_rpn(
    #    rpn_path="/home/huziy/skynet3_exec1/from_guillimin/Africa_044deg_geophy/pm1975120100_00000000p",
    #    outpath="africa_0.44deg_sand_clay_dpth.nc"
    #)
    #    delete_files_with_nrecords()
    #    extract_runoff_to_netcdf_folder(folder_path = 'data/CORDEX/NA_fix')

    #
    #    extract_field(name="VF", level=3, in_file="/home/huziy/skynet3_rech1/test/geophys_Quebec_86x86_0.5deg.v3")
    #    extract_field(name="VF", level=3, in_file="/home/huziy/skynet3_rech1/test/geophys_Quebec_260x260_3")


    #extract_sand_and_clay_from_rpn(rpn_path= "/b2_fs2/huziy/OMSC26_MPI_long_new_v01/geo_Arctic_198x186",
    #    outpath="/home/huziy/skynet3_rech1/runoff_arctic_nc/geo_Arctic_198x186.nc")

    # runoff_to_netcdf_parallel("/b2_fs2/huziy/Arctic_0.5deg_OMSC_26L_ERA40I/",
    #                           "/skynet3_rech1/huziy/runoff_arctic_nc/ERA40")

    # extract_sand_and_clay_from_rpn(
    #     rpn_path="/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/geophys_West_NA_0.25deg_104x75_GLNM_PRSF_CanHR85_SAND_CLAY_DPTH",
    #     outpath="/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/geophys_West_NA_0.25deg_104x75_GLNM_PRSF_CanHR85_sand_clay_dpth.nc")



    # For offline routing (CORDEX NA 0.11)
    # runoff_to_netcdf_parallel_with_multirpn("/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.11deg_links",
    #                                         "/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.11deg_NA_RUNOFF_NC")

    # extract_sand_and_clay_from_rpn(
    #     rpn_path="/HOME/data/Simulations/CRCM5/North_America/NorthAmerica_0.11deg_ERA40-Int0.75_B1/Samples/NorthAmerica_0.11deg_ERA40-Int0.75_B1_195801/pm1958010100_00000000p",
    #     outpath="/HOME/huziy/skynet3_rech1/water_route_mh_bc_011deg_wc/sand_clay_depth_to_bedrock_650x640.nc"
    # )

    # For offline routing (CORDEX NA 0.44)
    # runoff_to_netcdf_parallel_with_multirpn("/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.44deg_links",
    #                                         "/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.44deg_NA_RUNOFF_NC")
    #
    # extract_sand_and_clay_from_rpn(
    #     rpn_path="/HOME/data/Simulations/CRCM5/North_America/NorthAmerica_0.44deg_ERA40-Int0.75_B1/Samples/NorthAmerica_0.44deg_ERA40-Int0.75_B1_195801/pm1958010100_00000000p",
    #     outpath="/HOME/huziy/skynet3_rech1/water_route_mh_bc_044deg_wc/sand_clay_depth_to_bedrock_0.44_2012x200.nc"
    # )
    # -----------------------------------------

    # For offline routing (CORDEX NA 0.22)
    runoff_to_netcdf_parallel_with_multirpn("/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.22deg_links",
                                            "/RESCUE/skynet3_rech1/huziy/CORDEX_ERAI_0.22deg_NA_RUNOFF_NC")

    extract_sand_and_clay_from_rpn(
        rpn_path="/HOME/data/Simulations/CRCM5/North_America/NorthAmerica_0.22deg_ERA40-Int0.75_B1/Samples/NorthAmerica_0.22deg_ERA40-Int0.75_B1_195801/pm1958010100_00000000p",
        outpath="/HOME/huziy/skynet3_rech1/water_route_mh_bc_022deg_wc/sand_clay_depth_to_bedrock_0.22_380x360.nc"
    )

    print("Hello World")
