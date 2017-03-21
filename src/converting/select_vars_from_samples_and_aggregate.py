from collections import defaultdict
from multiprocessing.pool import Pool
from pathlib import Path

import xarray
from rpn import level_kinds
from rpn.rpn import RPN

# Mostly for 2D arrays in time
import numpy as np

from netCDF4 import date2num

integrated_wv_RPN_name = "IWVM"

rpn_name_to_nc_name = {
    "PR": "pr",
    integrated_wv_RPN_name: "prw",
    "STFA": "streamflow"
}


rpn_name_to_long_name = {
    "PR": "total precipitation",
    integrated_wv_RPN_name: "integrated water vapor",
    "STFA": "river discharge"
}

rpn_name_to_units = {
    "PR": "M/s",
    "IIRM": "kg/m**2",
    "ICRM": "kg/m**2",
    integrated_wv_RPN_name: "kg/m**2",
    "STFA": "m**3/s"
}


default_global_nc_attributes = {
    "prepared_by": "Oleksandr Huziy (guziy.sasha@gmail.com)",
    "organization": "ESCER/UQAM",
    "simulations_performed_by": "Oleksandr Huziy",
    "project": "Engage"
}


def get_default_mult():
    return 1

rpn_name_to_mult = defaultdict(get_default_mult)
rpn_name_to_mult["PR"] = 1.0e3,  # To convert M/s to kg /(m**2 * s)

rpn_name_to_ncunits = {
    "PR": "kg m-2 s-1",
    integrated_wv_RPN_name: "kg m-2",
    "STFA": "m3 s-1"
}



def extract_data_for_year_in_parallel(kwargs):
    try:
        extract_data_for_year(**kwargs)
    except Exception as e:
        print("Exception occurred in a process")
        print("inputs: ", kwargs)
        print(e)




def extract_data_for_year(year: int = 1980, varnames = None, samples_dir: Path = None, out_dir: Path = None,
                          level: int = -1, level_kind: int = level_kinds.ARBITRARY, target_freq_hours=6, calendar_str=None):

    month_folders = [mf for mf in samples_dir.iterdir() if mf.name[:-2].endswith(str(year))]

    # get the data from a month before in case there are some values from this year there
    month_folders += [mf for mf in samples_dir.iterdir() if mf.name.endswith("{}12".format(year - 1))]

    data_for_var = {}


    vname_to_outfile = {}

    for vname in varnames:
        vname_to_outfile[vname] = str(out_dir.joinpath("{}_{}.nc".format(rpn_name_to_nc_name[vname], year)))



    lons = None
    lats = None
    rlons = None
    rlats = None

    projparams = None

    for vname in varnames:

        # If the output file already exists, skip it
        if Path(vname_to_outfile[vname]).is_file():
            print("{} already exists, skipping ...".format(vname_to_outfile[vname]))
            continue


        data_for_var[vname] = {}

        for mf in month_folders:
            for data_file in mf.iterdir():
                # skip hidden files
                if data_file.name.startswith("."):
                    continue

                if data_file.name.endswith("~"):
                    continue

                try:
                    r = RPN(str(data_file))

                    vars_in_file = r.get_list_of_varnames()

                    if vname in vars_in_file:
                        data = r.get_all_time_records_for_name_and_level(varname=vname, level=level, level_kind=level_kind)
                        data_for_var[vname].update(data)


                        if lons is None:
                            projparams = r.get_proj_parameters_for_the_last_read_rec()
                            rlons, rlats = r.get_tictacs_for_the_last_read_record()
                            lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()

                except Exception:
                    pass

                finally:
                    r.close()


    # ---- Save retrieved data to netcdf
    if calendar_str is None:
        calendar_str = "proleptic_gregorian"

    for vname in data_for_var:
        print("Inside the loop over data_for_var")

        times = list(sorted(d for d in data_for_var[vname] if d.year == year))

        data = np.array([data_for_var[vname][d] for d in times])

        # change the order of dimensions and convert the units if required
        data = np.transpose(data, (0, 2, 1)) * rpn_name_to_mult[vname]
        
        print("calendar_str = {}".format(calendar_str))

        ds = xarray.Dataset({
                vname: (("time", "rlat", "rlon"), data, {"units": rpn_name_to_ncunits[vname], "long_name": rpn_name_to_long_name[vname]}),
            },
            coords = {
                "rlon": (["rlon"], rlons, {"long_name": "rotated longitudes"}),
                "rlat": (["rlat"], rlats, {"long_name": "rotated latitudes"}),
                "time": (["time"], times),
                "lon": (["rlat", "rlon"], lons.T, {"long_name": "geographic longitude", "ranges": "{} .. {}".format(lons.min(), lons.max())}),
                "lat": (["rlat", "rlon"], lats.T, {"long_name": "geographic latitude", "ranges": "{} .. {}".format(lats.min(), lats.max())}),
                "rotated_latlon": (("dummy", ), [int(0),], projparams)
            },
            attrs=default_global_nc_attributes
        )

        # "Calculate target_freq_hours-hourly means"
        ds = ds.resample("{}H".format(target_freq_hours), "time", keep_attrs=True, closed="left")


        # construct the name of the output file
        out_file = vname_to_outfile[vname]
        print("saving data to {}".format(out_file))


        # Rename the names of the variables if required
        ds.rename({vname: rpn_name_to_nc_name[vname]}, inplace=True)
        
        # Remove the all-nan fields, if any have creeped in during resampling
        ds = ds.dropna(dim="time", how="all")

        ds.to_netcdf(path=out_file, encoding={"time": {"dtype": "int32", "calendar": calendar_str}, "rotated_latlon": {"dtype": "int32"}})

    return 0


def main_era_interim():
    samples_dir_p = Path("/RECH/data/Simulations/CRCM5/North_America/NorthAmerica_0.44deg_ERA40-Int0.75_QC_B1/Samples")

    out_dir_root = Path("/RECH2/huziy/BenAlaya/")


    if samples_dir_p.name.lower() == "samples":
        out_folder_name = samples_dir_p.parent.name
    else:
        out_folder_name = samples_dir_p.name


    varnames = ["PR", integrated_wv_RPN_name]  # Total precipitation m/s; integrated ice, liquid water and vapor (in kg/m**2) averaged over last MOYHR

    # ======================================

    out_dir_p = out_dir_root.joinpath(out_folder_name)

    if not out_dir_p.is_dir():
        out_dir_p.mkdir()

    inputs = []
    for y in range(1989, 2010):
        inputs.append(dict(year=y, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6, calendar_str=None))

    # Extract the data for each year in parallel
    pool = Pool(processes=3)
    pool.map(extract_data_for_year_in_parallel, inputs)

    # extract_data_for_year(1980, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6)



def main_canesm2_historical():
    samples_dir_p = Path("/RECH/data/Simulations/CRCM5/North_America/NorthAmerica_0.44deg_CanHisto_B1/Samples")

    out_dir_root = Path("/RECH2/huziy/BenAlaya/")


    if samples_dir_p.name.lower() == "samples":
        out_folder_name = samples_dir_p.parent.name
    else:
        out_folder_name = samples_dir_p.name


    varnames = ["PR", integrated_wv_RPN_name]  # Total precipitation m/s; integrated ice, liquid water and vapor (in kg/m**2) averaged over last MOYHR

    # ======================================

    out_dir_p = out_dir_root.joinpath(out_folder_name)

    if not out_dir_p.is_dir():
        out_dir_p.mkdir()



    inputs = []
    for y in range(1951, 2006):
        inputs.append(dict(year=y, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6, calendar_str="365_day"))

    # Extract the data for each year in parallel
    pool = Pool(processes=5)
    pool.map(extract_data_for_year_in_parallel, inputs)

    # extract_data_for_year(1980, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6)



def main_canesm2_rcp45():
    samples_dir_p = Path("/RECH/data/Simulations/CRCM5/North_America/NorthAmerica_0.44deg_CanRCP45_B1/Samples")

    out_dir_root = Path("/RECH2/huziy/BenAlaya/")


    if samples_dir_p.name.lower() == "samples":
        out_folder_name = samples_dir_p.parent.name
    else:
        out_folder_name = samples_dir_p.name


    varnames = ["PR", integrated_wv_RPN_name]  # Total precipitation m/s; integrated ice, liquid water and vapor (in kg/m**2) averaged over last MOYHR

    # ======================================

    out_dir_p = out_dir_root.joinpath(out_folder_name)

    if not out_dir_p.is_dir():
        out_dir_p.mkdir()



    inputs = []
    for y in range(2006, 2101):
        inputs.append(dict(year=y, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6, calendar_str="365_day"))

    # Extract the data for each year in parallel
    pool = Pool(processes=3)
    pool.map(extract_data_for_year_in_parallel, inputs)

    # extract_data_for_year(1980, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6)


def main_canesm2_rcp85():
    samples_dir_p = Path("/RECH/data/Simulations/CRCM5/North_America/NorthAmerica_0.44deg_CanRCP85_B1/Samples")

    out_dir_root = Path("/RECH2/huziy/BenAlaya/")


    if samples_dir_p.name.lower() == "samples":
        out_folder_name = samples_dir_p.parent.name
    else:
        out_folder_name = samples_dir_p.name


    varnames = ["PR", integrated_wv_RPN_name]  # Total precipitation m/s; integrated ice, liquid water and vapor (in kg/m**2) averaged over last MOYHR

    # ======================================

    out_dir_p = out_dir_root.joinpath(out_folder_name)

    if not out_dir_p.is_dir():
        out_dir_p.mkdir()


    inputs = []
    for y in range(2006, 2101):
        inputs.append(dict(year=y, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6, calendar_str="365_day"))

    # Extract the data for each year in parallel
    pool = Pool(processes=3)
    pool.map(extract_data_for_year_in_parallel, inputs)

    # extract_data_for_year(1980, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6)


def main_mh():
    samples_dir_p = Path("/RECH2/huziy/BC-MH/bc_mh_044deg/Samples")

    out_dir_root = Path("/RECH2/huziy/MH_streamflows/")


    if samples_dir_p.name.lower() == "samples":
        out_folder_name = samples_dir_p.parent.name
    else:
        out_folder_name = samples_dir_p.name


    varnames = ["STFA", ]

    # ======================================

    out_dir_p = out_dir_root.joinpath(out_folder_name)

    if not out_dir_p.is_dir():
        out_dir_p.mkdir(parents=True)


    inputs = []
    for y in range(1981, 2010):
        inputs.append(dict(year=y, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=24))

    # Extract the data for each year in parallel
    pool = Pool(processes=3)
    pool.map(extract_data_for_year_in_parallel, inputs)

    # extract_data_for_year(1980, varnames=varnames, samples_dir=samples_dir_p, out_dir=out_dir_p, target_freq_hours=6)



if __name__ == '__main__':
    #main_era_interim()
    #main_canesm2_historical()
    #main_canesm2_rcp45()
    #main_canesm2_rcp85()

    from multiprocessing import Process

    # funcs = [main_era_interim, main_canesm2_historical, main_canesm2_rcp45, main_canesm2_rcp85]

    # funcs = [main_canesm2_historical, ]

    funcs = [main_mh, ]

    for i, f in enumerate(funcs):
        p = Process(target=f)
        p.start()




