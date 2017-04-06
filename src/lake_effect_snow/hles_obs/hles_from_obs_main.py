from collections import OrderedDict
from datetime import datetime, timedelta
from pendulum import Period
from rpn import level_kinds

from lake_effect_snow import data_source_types
from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import vname_to_offset_CRCM5, vname_to_multiplier_CRCM5, \
    vname_to_fname_prefix_CRCM5, T_AIR_2M, U_WE, V_SN, vname_map_CRCM5
from lake_effect_snow.lake_effect_snowfall_entry import calculate_lake_effect_snowfall_each_year_in_parallel


def main():
    label = "Obs"


    period = Period(
        datetime(1980, 11, 1), datetime(2009, 2, 1)
    )

    # should be consequent
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
            "base_folder": "/HOME/huziy/skynet3_rech1/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260",
            "data_source_type": data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
            "min_dt": timedelta(hours=3),
            "varname_mapping": vname_map,
            "level_mapping": vname_to_level_erai,
            "offset_mapping": vname_to_offset_CRCM5,
            "multiplier_mapping": vname_to_multiplier_CRCM5,
            "filename_prefix_mapping": vname_to_fname_prefix_CRCM5,
            "out_folder": "lake_effect_analysis_icefix_{}_{}-{}".format(label, period.start.year, period.end.year)
        }
    )])

    calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config=label_to_config, period=period, nprocs_to_use=15)


if __name__ == '__main__':
    main()
