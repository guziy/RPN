import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num
from rpn import level_kinds
from rpn.rpn_multi import MultiRPN

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler import commons
from crcm5.nemo_vs_hostetler import nemo_hl_util

# Method: proceed month by month:
#   a) read in tprecip
#   b) read in 2-m air temperature
#   c) Calculate snow fall using formulas from Notaro et al 2015.
from crcm5.nemo_vs_hostetler.nemo_hl_util import get_monthyeardate_to_paths_map

FRESH_SNOW_MIN_DENSITY_KG_PER_M3 = 50.0


def get_snow_fall_m_per_s(precip_m_per_s, tair_deg_c):
    """

    :param precip_m_per_s: total precipitation field in M/s
    :param tair_deg_c: 2-m air temperature in degC
    Formula is based on Notaro et al 2015
    returns snowfall in M/s - same as precip units

    Note: it is actual snow fall, not water equivalent
    """


    result = np.zeros_like(precip_m_per_s)
    rhos = np.zeros_like(precip_m_per_s)


    where_snow = tair_deg_c < 0

    # if there is no points below the freezing point
    if not np.any(where_snow):
        return result

    result[~where_snow] = 0.0

    rhos[where_snow & (tair_deg_c < -15)] = FRESH_SNOW_MIN_DENSITY_KG_PER_M3

    where_snow_and_very_cold = where_snow & (tair_deg_c >= -15)
    if np.any(where_snow_and_very_cold):
        rhos[where_snow_and_very_cold] = FRESH_SNOW_MIN_DENSITY_KG_PER_M3 + 1.7 * (tair_deg_c[where_snow_and_very_cold] + 15) ** 1.5

    result[where_snow] = precip_m_per_s[where_snow] * commons.water_density / rhos[where_snow]

    return result


@main_decorator
def main():
    start_year = 1979
    end_year = 1981

    HL_LABEL = "CRCM5_HL"
    NEMO_LABEL = "CRCM5_NEMO"

    file_prefix = "pm"
    PR_level = -1
    PR_level_type = level_kinds.ARBITRARY

    tprecip_vname = "PR"
    sprecip_vname = "SN"


    TT_level = 1
    TT_level_type = level_kinds.HYBRID



    sim_label_to_path = OrderedDict(
        [(HL_LABEL, "/RESCUE/skynet3_rech1/huziy/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl_oneway/Samples"),
         (NEMO_LABEL, "/HOME/huziy/skynet3_rech1/CNRCWP/C5/2016/2-year-runs/coupled-GL+stfl/Samples")]
    )

    # get a coord file ...
    coord_file = ""
    found_coord_file = False
    for mdir in os.listdir(sim_label_to_path[HL_LABEL]):

        mdir_path = os.path.join(sim_label_to_path[HL_LABEL], mdir)
        if not os.path.isdir(mdir_path):
            continue

        for fn in os.listdir(mdir_path):
            print(fn)
            if fn[:2] not in ["pm", "dm", "pp", "dp"]:
                continue

            coord_file = os.path.join(mdir_path, fn)
            found_coord_file = True

        if found_coord_file:
            break

    bmp, lons, lats = nemo_hl_util.get_basemap_obj_and_coords_from_rpn_file(path=coord_file)
    xx, yy = bmp(lons, lats)



    # read necessary input and calculate snowfall and save to the file
    for sim_label, samples_dir in sim_label_to_path.items():
        samples_dir_path = Path(samples_dir)

        TT_monthdate_to_paths = get_monthyeardate_to_paths_map(file_prefix="dm", start_year=start_year, end_year=end_year, samples_dir_path=samples_dir_path)
        PR_monthdate_to_paths = get_monthyeardate_to_paths_map(file_prefix="pm", start_year=start_year, end_year=end_year, samples_dir_path=samples_dir_path)



        with Dataset(str(samples_dir_path.parent / "{}_snow_fall_{}-{}.nc".format(sim_label, start_year, end_year)), "w") as ds:


            assert isinstance(ds, Dataset)


            ds.createDimension("time", None)
            ds.createDimension("lon", lons.shape[0])
            ds.createDimension("lat", lons.shape[1])




            # create the schema of the output file
            snow_fall = ds.createVariable(sprecip_vname, "f4", dimensions=("time", "lon", "lat"))
            lons_var = ds.createVariable("lon", "f4", dimensions=("lon", "lat"))
            lats_var = ds.createVariable("lat", "f4", dimensions=("lon", "lat"))
            time_var = ds.createVariable("time", "i8", dimensions=("time", ))
            time_var.units = "hours since {:%Y-%m-%d %H:%M:%S}".format(datetime(start_year, 1, 1))


            lons_var[:] = lons
            lats_var[:] = lats



            # use sorted dates
            record_count = 0
            for month_date in sorted(TT_monthdate_to_paths):

                tt = MultiRPN(path=TT_monthdate_to_paths[month_date]).get_all_time_records_for_name_and_level(varname="TT", level=TT_level, level_kind=TT_level_type)

                pr = MultiRPN(path=PR_monthdate_to_paths[month_date]).get_all_time_records_for_name_and_level(varname="PR", level=PR_level, level_kind=PR_level_type)

                print("Processing {}".format(month_date))
                for d in sorted(tt):
                    t_field = tt[d]
                    pr_field = pr[d]

                    sn = get_snow_fall_m_per_s(precip_m_per_s=pr_field, tair_deg_c=t_field)

                    snow_fall[record_count, :, :] = sn
                    time_var[record_count] = date2num(d, time_var.units)
                    record_count += 1








if __name__ == '__main__':
    main()