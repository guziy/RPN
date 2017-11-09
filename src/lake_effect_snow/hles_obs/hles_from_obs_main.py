from collections import OrderedDict
from datetime import datetime, timedelta

from pendulum import Period
from rpn import level_kinds

from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import vname_to_offset_CRCM5, vname_to_multiplier_CRCM5, \
    vname_to_fname_prefix_CRCM5, T_AIR_2M, U_WE, V_SN, vname_map_CRCM5
from lake_effect_snow.lake_effect_snowfall_entry import calculate_lake_effect_snowfall_each_year_in_parallel


def main():
    label = "Obs"


    period = Period(
        datetime(1980, 11, 1), datetime(1981, 2, 1)
    )

    # should be continuous??
    months_of_interest = [11, 12, 1]

    period.months_of_interest = months_of_interest


    vname_to_level_erai = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }



    vname_map = {}
    vname_map.update(vname_map_CRCM5)


    label_to_config = OrderedDict([(
        label, {
            DataManager.SP_BASE_FOLDER: "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260",
            DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
            DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
            DataManager.SP_LEVEL_MAPPING: vname_to_level_erai,
            DataManager.SP_OFFSET_MAPPING: vname_to_offset_CRCM5,
            DataManager.SP_MULTIPLIER_MAPPING: vname_to_multiplier_CRCM5,
            DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: vname_to_fname_prefix_CRCM5,
            "out_folder": "lake_effect_analysis_icefix_{}_{}-{}_test1".format(label, period.start.year, period.end.year)
        }
    )])

    calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config=label_to_config, period=period, nprocs_to_use=15)


if __name__ == '__main__':

    import time
    t0 = time.time()
    main()
    print(f"Execution time: {time.time() - t0} (s)")
