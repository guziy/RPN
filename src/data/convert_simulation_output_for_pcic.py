from collections import OrderedDict, defaultdict

from pendulum import Pendulum
from rpn import level_kinds

from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import vname_to_offset_CRCM5, vname_to_multiplier_CRCM5, \
    vname_to_fname_prefix_CRCM5, T_AIR_2M, U_WE, V_SN, vname_map_CRCM5

from copy import copy


def main():

    global_metadata = OrderedDict([
        ("source_dir", ""),
        ("project", "CNRCWP, NEI"),
        ("website", "http://cnrcwp.ca"),
        ("converted_on", Pendulum.now().to_day_datetime_string()),
    ])


    field_list = ["PR"]

    metadata = {
        "PR": {
            "long_name": "total precipitation",
            "units": "mm/day"
        }
    }

    offsets = copy(vname_to_offset_CRCM5)
    multipliers = copy(vname_to_multiplier_CRCM5)
    multipliers["PR"] = 1000 * 24 * 3600  # convert M/s to mm/day ()

    vname_to_fname_prefix = dict(vname_to_fname_prefix_CRCM5)
    vname_to_fname_prefix.update({
        "PR": "pm"
    })


    start_year = 1980
    end_year = 1981

    vname_to_level = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }

    vname_map = {}
    vname_map.update(vname_map_CRCM5)

    for vn in field_list:
        vname_map[vn] = vn


    label_to_simpath = OrderedDict()
    label_to_simpath["WC044_modified"] = "/snow3/huziy/NEI/WC/debug_NEI_WC0.44deg_Crr1/Samples"
    label_to_simpath["WC011_modified"] = "/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Samples"




    for label, simpath in label_to_simpath.items():

        global_metadata["source_dir"] = simpath

        store_config = {
            DataManager.SP_BASE_FOLDER: simpath,
            DataManager.SP_DATASOURCE_TYPE: data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
            DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
            DataManager.SP_LEVEL_MAPPING: vname_to_level,
            DataManager.SP_OFFSET_MAPPING: offsets,
            DataManager.SP_MULTIPLIER_MAPPING: multipliers,
            DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: vname_to_fname_prefix,
        }

        dm = DataManager(store_config=store_config)

        dm.export_to_netcdf(start_year=start_year, end_year=end_year,
                            field_names=field_list, label=label,
                            field_metadata=metadata, global_metadata=global_metadata)


if __name__ == '__main__':
    main()
