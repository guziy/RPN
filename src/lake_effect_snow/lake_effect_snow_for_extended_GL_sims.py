from collections import OrderedDict
from datetime import datetime, timedelta

from pendulum import Period
from rpn import level_kinds

from lake_effect_snow import data_source_types
from lake_effect_snow import default_varname_mappings
from lake_effect_snow.default_varname_mappings import vname_map_CRCM5, vname_to_offset_CRCM5, \
    vname_to_fname_prefix_CRCM5

from lake_effect_snow.default_varname_mappings import vname_to_multiplier_CRCM5

from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.lake_effect_snowfall_entry import calculate_lake_effect_snowfall, \
    calculate_lake_effect_snowfall_each_year_in_parallel


from lake_effect_snow.default_varname_mappings import T_AIR_2M, U_WE, V_SN, TOTAL_PREC, SNOWFALL_RATE


def main():
    period = Period(
        datetime(1979, 12, 1), datetime(1988, 3, 1)
    )

    # should be consequent
    months_of_interest = [12, 1, 2]

    period.months_of_interest = months_of_interest


    vname_to_level_erai = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }


    ERAI_label = "ERA-Interim"
    label = ERAI_label
    label_to_config = OrderedDict(
        [  # ERA-Interim
            (label,
             {
                 "base_folder": "/RECH/data/Driving_data/Offline/ERA-Interim_0.75/6h_Analysis",
                 "data_source_type": data_source_types.ALL_VARS_IN_A_FOLDER_OF_RPN_FILES,
                 "min_dt": timedelta(hours=6),
                 "varname_mapping": vname_map_CRCM5,
                 "level_mapping": vname_to_level_erai,
                 "offset_mapping": vname_to_offset_CRCM5,
                 "multiplier_mapping": vname_to_multiplier_CRCM5,
                 "out_folder": "lake_effect_analysis_{}_{}-{}".format(label, period.start.year, period.end.year)
             }
             ),
            # Add additional sources below
        ]
    )

    # calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config=label_to_config, period=period)
    # calculate_lake_effect_snowfall(label_to_config=label_to_config, period=period)



    label = "CRCM5_NEMO"
    label_to_config_CRCM5 = OrderedDict([(
        label, {
            "base_folder": "/RECH2/huziy/coupling/coupled-GL-NEMO1h_30min/Samples",
            "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
            "min_dt": timedelta(hours=3),
            "varname_mapping": vname_map_CRCM5,
            "level_mapping": vname_to_level_erai,
            "offset_mapping": vname_to_offset_CRCM5,
            "multiplier_mapping": vname_to_multiplier_CRCM5,
            "filename_prefix_mapping": vname_to_fname_prefix_CRCM5,
            "out_folder": "lake_effect_analysis_{}_{}-{}".format(label, period.start.year, period.end.year)
        }
    )])

#    calculate_lake_effect_snowfall(label_to_config=label_to_config_CRCM5, period=period)
    # calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config=label_to_config_CRCM5, period=period)



    label = "CRCM5_Hostetler"
    label_to_config_CRCM5 = OrderedDict([(
        label, {
            "base_folder": "/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected",
            "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME,
            "min_dt": timedelta(hours=3),
            "varname_mapping": vname_map_CRCM5,
            "level_mapping": vname_to_level_erai,
            "offset_mapping": vname_to_offset_CRCM5,
            "multiplier_mapping": vname_to_multiplier_CRCM5,
            "filename_prefix_mapping": vname_to_fname_prefix_CRCM5,
            "out_folder": "lake_effect_analysis_{}_{}-{}".format(label, period.start.year, period.end.year)
        }
    )])

    calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config=label_to_config_CRCM5, period=period)
#    calculate_lake_effect_snowfall(label_to_config=label_to_config_CRCM5, period=period)


if __name__ == '__main__':
    import time
    t0 = time.clock()
    main()
    print("Execution time {} seconds".format(time.clock() - t0))
