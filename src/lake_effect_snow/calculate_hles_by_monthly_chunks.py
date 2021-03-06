from collections import OrderedDict
from datetime import datetime, timedelta
from multiprocessing.pool import Pool

from pendulum import Period
from rpn import level_kinds

from application_properties import main_decorator
from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import vname_map_CRCM5, T_AIR_2M, U_WE, V_SN, vname_to_offset_CRCM5, \
    vname_to_multiplier_CRCM5, vname_to_fname_prefix_CRCM5
from lake_effect_snow.lake_effect_snowfall_entry import calculate_lake_effect_snowfall_each_year_in_parallel


# Calculate monthly HLES

def monthly_func(x):
    return calculate_lake_effect_snowfall_each_year_in_parallel(**x)


@main_decorator
def main_obs():
    label = "Obs_monthly_icefix_test2_1proc_speedtest_3"


    period = Period(
        datetime(1980, 1, 1), datetime(2010, 12, 31)
    )


    pool = Pool(processes=20)

    input_params = []
    for month_start in period.range("months"):

        month_end = month_start.add(months=1).subtract(seconds=1)

        current_month_period = Period(month_start, month_end)
        current_month_period.months_of_interest = [month_start.month, ]


        vname_to_level_erai = {
            T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
            U_WE: VerticalLevel(1, level_kinds.HYBRID),
            V_SN: VerticalLevel(1, level_kinds.HYBRID),
        }

        vname_map = {}
        vname_map.update(vname_map_CRCM5)

        label_to_config = OrderedDict([(
            label, {
                DataManager.SP_BASE_FOLDER: "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260_icefix",
                DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
                DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
                DataManager.SP_LEVEL_MAPPING: vname_to_level_erai,
                DataManager.SP_OFFSET_MAPPING: vname_to_offset_CRCM5,
                DataManager.SP_MULTIPLIER_MAPPING: vname_to_multiplier_CRCM5,
                DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: vname_to_fname_prefix_CRCM5,
                "out_folder": "lake_effect_analysis_daily_{}_{}-{}".format(label, period.start.year, period.end.year)
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
def main_crcm5_nemo():
    label = "CRCM5_NEMO"

    period = Period(
        datetime(1980, 1, 1), datetime(2015, 12, 31)
    )


    pool = Pool(processes=10)

    input_params = []
    for month_start in period.range("months"):

        month_end = month_start.add(months=1).subtract(seconds=1)

        current_month_period = Period(month_start, month_end)
        current_month_period.months_of_interest = [month_start.month, ]


        vname_to_level_erai = {
            T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
            U_WE: VerticalLevel(1, level_kinds.HYBRID),
            V_SN: VerticalLevel(1, level_kinds.HYBRID),
        }



        vname_map = {}
        vname_map.update(vname_map_CRCM5)

        vname_map = {}
        vname_map.update(vname_map_CRCM5)
        vname_map.update({
            default_varname_mappings.SNOWFALL_RATE: "SN"
        })

        label_to_config = OrderedDict([(
            label, {
                DataManager.SP_BASE_FOLDER: "/snow3/huziy/NEI/GL/erai0.75deg_driven/GL_with_NEMO_dtN_1h_and_30min/Samples",
                DataManager.SP_DATASOURCE_TYPE: data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
                DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
                DataManager.SP_LEVEL_MAPPING: vname_to_level_erai,
                DataManager.SP_OFFSET_MAPPING: vname_to_offset_CRCM5,
                DataManager.SP_MULTIPLIER_MAPPING: vname_to_multiplier_CRCM5,
                DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: default_varname_mappings.vname_to_fname_prefix_CRCM5,
                "out_folder": "lake_effect_analysis_{}_{}-{}_monthly".format(label, period.start.year, period.end.year)
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
def main_crcm5_hl():
    label = "CRCM5_HL"

    period = Period(
        datetime(1980, 1, 1), datetime(2009, 12, 31)
    )


    pool = Pool(processes=12)

    input_params = []
    for month_start in period.range("months"):

        month_end = month_start.add(months=1).subtract(seconds=1)

        current_month_period = Period(month_start, month_end)
        current_month_period.months_of_interest = [month_start.month, ]


        vname_to_level_erai = {
            T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
            U_WE: VerticalLevel(1, level_kinds.HYBRID),
            V_SN: VerticalLevel(1, level_kinds.HYBRID),
        }

        vname_map = {}
        vname_map.update(vname_map_CRCM5)

        vname_map = {}
        vname_map.update(vname_map_CRCM5)
        vname_map.update({
            default_varname_mappings.SNOWFALL_RATE: "U3"
        })

        label_to_config = OrderedDict([(
            label, {
                DataManager.SP_BASE_FOLDER: "/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected",
                DataManager.SP_DATASOURCE_TYPE: data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME,
                DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
                DataManager.SP_LEVEL_MAPPING: vname_to_level_erai,
                DataManager.SP_OFFSET_MAPPING: vname_to_offset_CRCM5,
                DataManager.SP_MULTIPLIER_MAPPING: vname_to_multiplier_CRCM5,
                "out_folder": "lake_effect_analysis_{}_{}-{}_monthly".format(label, period.start.year, period.end.year)
            }
        )])

        kwargs = dict(
            label_to_config=label_to_config,
            period=current_month_period,
            months_of_interest=current_month_period.months_of_interest,
            nprocs_to_use=1
        )

        print(current_month_period.months_of_interest)
        input_params.append(kwargs)

    # execute in parallel
    pool.map(monthly_func, input_params)


if __name__ == '__main__':
    import time
    # main_crcm5_nemo()
    # main_crcm5_hl()
    t0 = time.clock()
    main_obs()
    print(f"Elapsed time: {time.clock() - t0} (s)")
