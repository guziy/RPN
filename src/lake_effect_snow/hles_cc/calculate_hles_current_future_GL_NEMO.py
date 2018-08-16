"""
This script uses the latest corrected coupled CRCM5_NEMO simulations.

"""




import matplotlib

from data.robust.data_manager import DataManager

matplotlib.use("Agg")

from collections import OrderedDict
from datetime import datetime, timedelta

from multiprocessing.pool import Pool

from pendulum import Period
from rpn import level_kinds

from application_properties import main_decorator
from lake_effect_snow import default_varname_mappings
from data.robust import data_source_types
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import vname_map_CRCM5, T_AIR_2M, U_WE, V_SN, vname_to_offset_CRCM5, \
    vname_to_multiplier_CRCM5, vname_to_fname_prefix_CRCM5
from lake_effect_snow.lake_effect_snowfall_entry import calculate_lake_effect_snowfall_each_year_in_parallel

# Calculate monthly HLES

def monthly_func(x):
    print(x)
    return calculate_lake_effect_snowfall_each_year_in_parallel(**x)


@main_decorator
def main_current(nprocs=20):

    period = Period(
        datetime(1989, 1, 1), datetime(2010, 12, 31)
    )

    label = "CRCM5_NEMO_fix_TT_PR_CanESM2_RCP85_{}-{}_monthly".format(period.start.year, period.end.year)

    vname_to_level_erai = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }


    vname_map = {}
    vname_map.update(vname_map_CRCM5)
    # vname_map[default_varname_mappings.SNOWFALL_RATE] = "SN"
    vname_map[default_varname_mappings.SNOWFALL_RATE] = "XXX"

    base_folder = "/scratch/huziy/Output/GL_CC_CanESM2_RCP85/coupled-GL-current_CanESM2/Samples"

    pool = Pool(processes=nprocs)

    input_params = []
    for month_start in period.range("months"):

        month_end = month_start.add(months=1).subtract(seconds=1)

        current_month_period = Period(month_start, month_end)
        current_month_period.months_of_interest = [month_start.month, ]

        label_to_config = OrderedDict([(
            label, {
                DataManager.SP_BASE_FOLDER: base_folder,
                DataManager.SP_DATASOURCE_TYPE: data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
                DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
                DataManager.SP_LEVEL_MAPPING: vname_to_level_erai,
                DataManager.SP_OFFSET_MAPPING: vname_to_offset_CRCM5,
                DataManager.SP_MULTIPLIER_MAPPING: vname_to_multiplier_CRCM5,
                DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: vname_to_fname_prefix_CRCM5,
                "out_folder": "lake_effect_analysis_{}_{}-{}".format(label, period.start.year, period.end.year)
            }
        )])

        kwargs = dict(
            label_to_config=label_to_config, period=current_month_period, months_of_interest=current_month_period.months_of_interest, nprocs_to_use=1
        )

        print(current_month_period.months_of_interest)
        input_params.append(kwargs)

    # execute in parallel
    pool.map(monthly_func, input_params)



@main_decorator
def main_future(nprocs=20):

    period = Period(
        datetime(2079, 1, 1), datetime(2100, 12, 31)
    )

    label = "CRCM5_NEMO_fix_TT_PR_CanESM2_RCP85_{}-{}_monthly".format(period.start.year, period.end.year)

    vname_to_level_erai = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }

    base_folder = "/scratch/huziy/Output/GL_CC_CanESM2_RCP85/coupled-GL-future_CanESM2/Samples"

    vname_map = {}
    vname_map.update(vname_map_CRCM5)
    # vname_map[default_varname_mappings.SNOWFALL_RATE] = "SN"
    vname_map[default_varname_mappings.SNOWFALL_RATE] = "XXX"



    pool = Pool(processes=nprocs)

    input_params = []
    for month_start in period.range("months"):

        month_end = month_start.add(months=1).subtract(seconds=1)

        current_month_period = Period(month_start, month_end)
        current_month_period.months_of_interest = [month_start.month, ]

        label_to_config = OrderedDict([(
            label, {
                # "base_folder": "/HOME/huziy/skynet3_rech1/CRCM5_outputs/cc_canesm2_rcp85_gl/coupled-GL-future_CanESM2/Samples",
                DataManager.SP_BASE_FOLDER: base_folder,
                DataManager.SP_DATASOURCE_TYPE: data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
                DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
                DataManager.SP_LEVEL_MAPPING: vname_to_level_erai,
                DataManager.SP_OFFSET_MAPPING: vname_to_offset_CRCM5,
                DataManager.SP_MULTIPLIER_MAPPING: vname_to_multiplier_CRCM5,
                DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: vname_to_fname_prefix_CRCM5,
                "out_folder": "lake_effect_analysis_{}_{}-{}".format(label, period.start.year, period.end.year)
            }
        )])

        kwargs = dict(
            label_to_config=label_to_config, period=current_month_period, months_of_interest=current_month_period.months_of_interest, nprocs_to_use=1
        )

        print(current_month_period.months_of_interest)
        input_params.append(kwargs)

    # execute in parallel
    pool.map(monthly_func, input_params)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:

        nprocs = 20
        if len(sys.argv) > 2:
           nprocs = int(sys.argv[2])

        if sys.argv[1].strip().lower() == "future":
            main_future(nprocs=nprocs)
        else:
            main_current(nprocs=nprocs)
    else:
        main_current()
