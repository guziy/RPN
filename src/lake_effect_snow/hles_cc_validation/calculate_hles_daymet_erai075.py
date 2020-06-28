from collections import OrderedDict
from multiprocessing.pool import Pool

from pendulum import datetime, Period
from rpn import level_kinds

from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.calculate_hles_by_monthly_chunks import monthly_func
from lake_effect_snow.default_varname_mappings import T_AIR_2M, V_SN, U_WE, vname_map_CRCM5, vname_to_offset_CRCM5, \
    vname_to_multiplier_CRCM5, vname_to_fname_prefix_CRCM5


def calculate_hles_daymet_erai075(nprocs=6):
    """
    Calculate HLES
    """
    # used for paper
    period = Period(
        datetime(1989, 1, 1), datetime(2010, 12, 31)
    )

    # debug
    # period = Period(
    #     datetime(1989, 1, 1), datetime(1990, 2, 1)
    # )

    label = "HLES_obs_daymet_erai075_niccis_{}-{}_based_on_452x260_v002".format(period.start.year, period.end.year)

    vname_to_level_erai = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }

    vname_to_multiplier = vname_to_multiplier_CRCM5.copy()

    # It seems that the Daymet data I have are in mm/day
    # converting to M/S
    vname_to_multiplier[default_varname_mappings.TOTAL_PREC] = 1e-3 / (24 * 3600)
    # vname_to_multiplier[default_varname_mappings.TOTAL_PREC] = 1 # debug test with anuspmauer tt,pr

    vname_map = {}
    vname_map.update(vname_map_CRCM5)
    # vname_map[default_varname_mappings.SNOWFALL_RATE] = "SN"
    vname_map[default_varname_mappings.SNOWFALL_RATE] = "XXX"

    base_folder = "/home/huziy/data/big1/Projects/observations/obs_data_for_HLES/interploated_to_the_same_grid/test"
    # base_folder = "/home/huziy/data/big1/Projects/observations/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_412x220_icefix_daymet"

    pool = Pool(processes=nprocs)

    input_params = []
    for month_count, month_start in enumerate(period.range("months")):
        month_end = month_start.add(months=1).subtract(seconds=1)

        current_month_period = Period(month_start, month_end)
        current_month_period.months_of_interest = [month_start.month, ]

        label_to_config = OrderedDict([(
            label, {
                DataManager.SP_BASE_FOLDER: base_folder,
                DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
                DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
                DataManager.SP_LEVEL_MAPPING: vname_to_level_erai,
                DataManager.SP_OFFSET_MAPPING: vname_to_offset_CRCM5,
                DataManager.SP_MULTIPLIER_MAPPING: vname_to_multiplier,
                DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: vname_to_fname_prefix_CRCM5,
                "out_folder": "{}_{}-{}".format(label, period.start.year, period.end.year)
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
    # for ip in input_params:
    #     monthly_func(ip)


if __name__ == '__main__':
    calculate_hles_daymet_erai075()