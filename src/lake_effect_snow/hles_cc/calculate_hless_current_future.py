import matplotlib
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
def main_current():

    period = Period(
        datetime(1989, 1, 1), datetime(2010, 12, 31)
    )

    label = "CRCM5_NEMO_CanESM2_RCP85_{}-{}".format(period.start.year, period.end.year)

    vname_to_level_erai = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }


    vname_map = {}
    vname_map.update(vname_map_CRCM5)
    vname_map[default_varname_mappings.SNOWFALL_RATE] = "SN"



    pool = Pool(processes=20)

    input_params = []
    for month_start in period.range("months"):

        month_end = month_start.add(months=1).subtract(seconds=1)

        current_month_period = Period(month_start, month_end)
        current_month_period.months_of_interest = [month_start.month, ]

        label_to_config = OrderedDict([(
            label, {
                #"base_folder": "/HOME/huziy/skynet3_rech1/CRCM5_outputs/cc_canesm2_rcp85_gl/coupled-GL-current_CanESM2/Samples",
                "base_folder": "/snow3/huziy/NEI/GL/GL_CC_CanESM2_RCP85/coupled-GL-current_CanESM2/Samples",
                "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
                "min_dt": timedelta(hours=3),
                "varname_mapping": vname_map,
                "level_mapping": vname_to_level_erai,
                "offset_mapping": vname_to_offset_CRCM5,
                "multiplier_mapping": vname_to_multiplier_CRCM5,
                "varname_to_filename_prefix_mapping": vname_to_fname_prefix_CRCM5,
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
def main_future():

    period = Period(
        datetime(2079, 1, 1), datetime(2100, 12, 31)
    )

    label = "CRCM5_NEMO_CanESM2_RCP85_{}-{}".format(period.start.year, period.end.year)

    vname_to_level_erai = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }


    vname_map = {}
    vname_map.update(vname_map_CRCM5)
    vname_map[default_varname_mappings.SNOWFALL_RATE] = "SN"



    pool = Pool(processes=20)

    input_params = []
    for month_start in period.range("months"):

        month_end = month_start.add(months=1).subtract(seconds=1)

        current_month_period = Period(month_start, month_end)
        current_month_period.months_of_interest = [month_start.month, ]

        label_to_config = OrderedDict([(
            label, {
                # "base_folder": "/HOME/huziy/skynet3_rech1/CRCM5_outputs/cc_canesm2_rcp85_gl/coupled-GL-future_CanESM2/Samples",
                "base_folder": "/snow3/huziy/NEI/GL/GL_CC_CanESM2_RCP85/coupled-GL-future_CanESM2/Samples",
                "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
                "min_dt": timedelta(hours=3),
                "varname_mapping": vname_map,
                "level_mapping": vname_to_level_erai,
                "offset_mapping": vname_to_offset_CRCM5,
                "multiplier_mapping": vname_to_multiplier_CRCM5,
                "varname_to_filename_prefix_mapping": vname_to_fname_prefix_CRCM5,
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
    # main_current()
    main_future()
