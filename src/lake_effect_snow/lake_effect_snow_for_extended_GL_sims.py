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
        datetime(1980, 12, 1), datetime(1985, 3, 1)
    )

    # should be consequent
    months_of_interest = [11, 12, 1]

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

    import time
    t0 = time.time()
    calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config=label_to_config, period=period, nprocs_to_use=5)
    print("Execution time: {} s".format(time.time() - t0))
    #calculate_lake_effect_snowfall(label_to_config=label_to_config, period=period)



    label = "CRCM5_NEMO"
    vname_map = {}
    vname_map.update(vname_map_CRCM5)
    vname_map.update({
        default_varname_mappings.SNOWFALL_RATE: "U3"
    })

    label_to_config_CRCM5 = OrderedDict([(
        label, {
            "base_folder": "/HOME/huziy/skynet3_rech1/CRCM5_outputs/coupled-GL-NEMO1h/selected_fields",
            "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME,
            "min_dt": timedelta(hours=3),
            "varname_mapping": vname_map,
            "level_mapping": vname_to_level_erai,
            "offset_mapping": vname_to_offset_CRCM5,
            "multiplier_mapping": vname_to_multiplier_CRCM5,
            "out_folder": "lake_effect_analysis_{}_{}-{}".format(label, period.start.year, period.end.year)
        }
    )])

    # calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config=label_to_config_CRCM5,
    #                                                      period=period,
    #                                                      nprocs_to_use=16)



    label = "CRCM5_Hostetler"

    vname_map = {}
    vname_map.update(vname_map_CRCM5)
    vname_map.update({
        default_varname_mappings.SNOWFALL_RATE: "U3"
    })

    label_to_config_CRCM5 = OrderedDict([(
        label, {
            "base_folder": "/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected",
            "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME,
            "min_dt": timedelta(hours=3),
            "varname_mapping": vname_map,
            "level_mapping": vname_to_level_erai,
            "offset_mapping": vname_to_offset_CRCM5,
            "multiplier_mapping": vname_to_multiplier_CRCM5,
            "filename_prefix_mapping": vname_to_fname_prefix_CRCM5,
            "out_folder": "lake_effect_analysis_{}_{}-{}".format(label, period.start.year, period.end.year)
        }
    )])

    # calculate_lake_effect_snowfall_each_year_in_parallel(label_to_config=label_to_config_CRCM5,
    #                                                      period=period,
    #                                                      nprocs_to_use=20)


if __name__ == '__main__':
    import time
    t0 = time.clock()
    main()
    print("Execution time {} seconds".format(time.clock() - t0))
