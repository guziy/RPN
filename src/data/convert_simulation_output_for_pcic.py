from collections import OrderedDict, defaultdict
import pendulum

from rpn import level_kinds

from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import vname_to_offset_CRCM5, vname_to_multiplier_CRCM5, \
    vname_to_fname_prefix_CRCM5, T_AIR_2M, U_WE, V_SN, vname_map_CRCM5

from copy import copy


def get_tops_and_bots_of_soil_layers(layer_widths):
    top = 0
    tops = []
    bots = []

    for w in layer_widths:
        tops.append(top)
        top += w
        bots.append(top)

    return {"z_top": tops, "z_bot": bots}




def parallel_conversion_entry():

    fields = ["PR", "AD", "AV", "GIAC",
                      "GIML", "GLD", "GLF", "GSAB",
                      "GSAC", "GSML", "GVOL", "GWDI",
                      "GWST", "GZ", "HR", "HU", "I1", "I2", "I4",
                      "I5", "MS", "N3", "N4", "P0", "PN", "PR", "S6", "SD",
                      "STFL", "SWSL", "SWSR", "T5", "T9", "TDRA", "TJ", "TRAF", "UD", "VD"]

    start_year = 1980
    end_year = 1981

    input = [[start_year, end_year, fname] for fname in fields]

    from multiprocessing import Pool

    p = Pool(processes=10)

    # do the conversion in parallel
    p.map(main_for_parallel_processing, input)



def main_for_parallel_processing(params):
    """
    :param params: list [start_year, end_year, field_name]
    """

    merge_chunks = False
    label_to_simpath = None
    if len(params) == 3:
        ys, ye, field_name = params
    elif len(params) == 4:
        ys, ye, field_name, label_to_simpath = params
    elif len(params) == 5:
        ys, ye, field_name, label_to_simpath, merge_chunks = params
    else:
        raise ValueError(f"Incorrect number of params passed to main_for_parallel_processing: {len(params)}")

    main(field_list=[field_name], start_year=ys, end_year=ye, label_to_simpath=label_to_simpath,
         merge_chunks=merge_chunks)


def main(field_list=None, start_year=1980, end_year=2010, label_to_simpath=None,
         merge_chunks=False):
    global_metadata = OrderedDict([
        ("source_dir", ""),
        ("project", "CNRCWP, NEI"),
        ("website", "http://cnrcwp.ca"),
        ("converted_on", pendulum.now().to_day_datetime_string()),
    ])


    if field_list is None:
        field_list = ["PR", "AD", "AV", "GIAC",
                      "GIML", "GLD", "GLF", "GSAB",
                      "GSAC", "GSML", "GVOL", "GWDI",
                      "GWST", "GZ", "HR", "HU", "I1", "I2", "I4",
                      "I5", "MS", "N3", "N4", "P0", "PN", "S6", "SD",
                      "STFL", "SWSL", "SWSR", "T5", "T9", "TDRA", "TJ", "TRAF", "UD", "VD"]

    fields_4d = field_list


    soil_level_widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                         1.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]


    subgrid_regions_levels = "lev=1: soil; lev=2: glacier; lev=3: water; lev=4:sea ice; lev=5: aggregated; lev=6: urban; lev=7: lakes."


    metadata = {
        "PR": {
            "long_name": "total precipitation",
            "units": "mm/day",
            "description": "total precipitation"
        },
        "AD": {
            "units": "W/m**2",
            "description": "ACCUMULATION OF FDSI(IR ENERGY FLUX TOWARDS GROUND)"
        },
        "AV": {
            "units": "W/m**2",
            "description": "ACCUMULATION OF FV(SURFACE LATENT FLUX)"
        },
        "DN": {
            "units": "kg/m**3",
            "description": "SNOW DENSITY"
        },
        "FN": {
            "description": "TOTAL CLOUDS"
        },
        "GIAC": {"units": "mm weq/s", "description": "ACCUMUL. OF GLACIER ICE ACCUMULATION [MM WEQ/S]"},
        "GIML": {"units": "mm weq/s", "description": "ACCUMUL. OF GLACIER ICE MELT [MM WEQ/S]"},
        "GLD": {"units": "m", "description": "MEAN GLACIER DEPTH FOR WHOLE GRID BOX [M ICE]"},
        "GLF": {"units": "", "description": "GLACIER FRACTION WRT WHOLE GRID"},
        "GSAB": {"units": "mm weq/s", "description": "ACCUMUL. OF SNOW ABLATION ON GLACIER [MM WEQ/S]"},
        "GSAC": {"units": "mm weq/s", "description": "ACCUMUL. OF SNOW ACCUMUL. ON GLACIER [MM WEQ/S]"},
        "GSML": {"units": "mm weq/s", "description": "ACCUMUL. OF SNOW MELT ON GLACIER [MM WEQ/S]"},
        "GVOL": {"units": "m**3 ice", "description": "GLACIER VOLUME FOR WHOLE GRID BOX [M3 ICE]"},
        "GWDI": {"units": "m**3/s", "description": "GROUND WATER DISCHARGE , M**3/S"},
        "GWST": {"units": "m**3", "description": "GROUND WATER STORE , M**3"},
        "GZ": {"units": "dam", "description": "GEOPOTENTIAL HEIGHT"},
        "HR": {"units": "", "description": "RELATIVE HUMIDITY"},
        "HU": {"units": "kg/kg", "description": "SPECIFIC HUMIDITY"},
        "I1": {"units": "m**3/m**3", "description": "SOIL VOLUMETRIC WATER CONTENTS"},
        "I2": {"units": "m**3/m**3", "description": "SOIL VOLUMETRIC ICE CONTENTS"},
        "I4": {"units": "kg/m**2", "description": "WATER IN THE SNOW PACK"},
        "I5": {"units": "kg/m**2", "description": "SNOW MASS"},
        "MS": {"units": "kg/(m**2 * s)", "description": "MELTING SNOW FROM SNOWPACK"},
        "N3": {"units": "mm/day",
               "description": "ACCUM. OF SOLID PRECIP. USED BY LAND SURFACE SCHEMES (LAGGS 1 TIME STEP FROM PR)"},
        "N4": {"units": "W/m**2", "description": "ACCUM. OF SOLAR RADATION"},
        "P0": {"units": "hPa", "description": "SURFACE PRESSURE"},
        "PN": {"units": "hPa", "description": "SEA LEVEL PRESSURE"},
        "S6": {"units": "", "description": "FRACTIONAL COVERAGE FOR SNOW"},
        "SD": {"units": "cm", "description": "SNOW DEPTH"},
        "STFL": {"units": "m**3/s", "description": "SURF. WATER STREAMFLOW IN M**3/S"},
        "SWSL": {"units": "m**3", "description": "SURF. WATER STORE (LAKE), M**3"},
        "SWSR": {"units": "m**3", "description": "SURF. WATER STORE (RIVER), M**3"},
        "T5": {"units": "K", "description": "MIN TEMPERATURE OVER LAST 24.0 HRS"},
        "T9": {"units": "K", "description": "MAX TEMPERATURE OVER LAST 24.0 HRS"},
        "TDRA": {"units": "kg/(m**2 * s)", "description": "ACCUM. OF BASE DRAINAGE"},
        "TJ": {"units": "K", "description": "SCREEN LEVEL TEMPERATURE"},
        "TRAF": {"units": "kg/(m**2 * s)", "description": "ACCUM. OF TOTAL SURFACE RUNOFF"},
        "UD": {"units": "knots", "description": "SCREEN LEVEL X-COMPONENT OF WIND"},
        "VD": {"units": "knots", "description": "SCREEN LEVEL Y-COMPONENT OF WIND"},
        "TT": {"units": "degC", "description": "Air temperature"}
    }


    # add descriptions of subgrid fraction levels
    for v in metadata:
        if v in ["TRAF", "TDRA", "SD"]:
            metadata[v]["description"] += ", " + subgrid_regions_levels

    soil_levels_map = get_tops_and_bots_of_soil_layers(soil_level_widths)
    vname_to_soil_layers = {"I1": soil_levels_map, "I2":soil_levels_map}


    offsets = copy(vname_to_offset_CRCM5)
    multipliers = copy(vname_to_multiplier_CRCM5)
    multipliers["PR"] = 1000 * 24 * 3600  # convert M/s to mm/day ()
    multipliers["N3"] = multipliers["PR"]  # M/s to mm/day

    vname_to_fname_prefix = dict(vname_to_fname_prefix_CRCM5)
    vname_to_fname_prefix.update({
        "PR": "pm",
        "HU": "dp",
        "HR": "dp",
        "GZ": "dp",
        "P0": "dm",
        "PN": "dm",
        "TT": "dm",
        "SN": "pm"
    })

    for vn in field_list:
        if vn not in vname_to_fname_prefix:
            vname_to_fname_prefix[vn] = "pm"


    vname_to_level = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }

    vname_map = {}
    vname_map.update(vname_map_CRCM5)

    for vn in field_list:
        vname_map[vn] = vn

    if label_to_simpath is None:
        label_to_simpath = OrderedDict()
        label_to_simpath["WC044_modified"] = "/snow3/huziy/NEI/WC/debug_NEI_WC0.44deg_Crr1/Samples"
        #label_to_simpath["WC011_modified"] = "/snow3/huziy/NEI/WC/NEI_WC0.11deg_Crr1/Samples"

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
                            field_metadata=metadata, global_metadata=global_metadata,
                            field_to_soil_layers=vname_to_soil_layers,
                            merge_chunks=merge_chunks)


if __name__ == '__main__':
    # main()
    parallel_conversion_entry()
