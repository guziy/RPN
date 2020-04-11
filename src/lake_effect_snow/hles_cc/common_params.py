from collections import defaultdict

import cartopy
from pathlib import Path
import numpy as np
from cartopy.feature import NaturalEarthFeature

# data_root = Path("/Users/huziy/HLES")
# data_root = Path("/Users/huziy/Projects")
data_root = Path("/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/HLES_analysis_air_backup/HLES_data_paper/")

img_folder = Path("hles_cc_paper")
from lake_effect_snow import default_varname_mappings

varname_to_display_name = {
    "lake_ice_fraction": "Lake ice fraction",
    "hles_snow": "HLES"
}


crcm_nemo_cur_label = "GEM_NEMOc"
crcm_nemo_fut_label = "GEM_NEMOf"


image_file_options = dict(
    bbox_inches="tight", dpi=300, transparent=True
)


LAKES_50m = NaturalEarthFeature('physical', 'lakes', '50m')
COASTLINE_50m = NaturalEarthFeature('physical', 'coastline', '50m')

OCEAN_50m = NaturalEarthFeature('physical', 'ocean', '50m')


RIVERS_50m = NaturalEarthFeature("physical", "rivers_lake_centerlines", "50m")


var_display_names = {
    "hles_snow": "HLES",
    "hles_snow_days": "HLES freq",
    "lake_ice_fraction": "Lake ice fraction",
    "TT": "2m air\n temperature",
    "PR": "total\nprecipitation",
    "cao_days": "CAO\nfrequency",
    default_varname_mappings.HLES_AMOUNT: "HLES",
    default_varname_mappings.LAKE_ICE_FRACTION: "Lake ice\nfraction",
    default_varname_mappings.T_AIR_2M: "2m air\n temperature",
    default_varname_mappings.TOTAL_PREC: "total\nprecipitation",
    default_varname_mappings.HLES_FREQUENCY: "HLES\nfrequency",
    default_varname_mappings.CAO: "CAO\nfrequency",
    default_varname_mappings.SNOWFALL_RATE: "Snowfall"
}

# add display names for biases
bias_display_names = {
    f"bias_{key}": f"Bias: {value}" for key, value in var_display_names.items()
}
var_display_names.update(bias_display_names)


bias_vname_to_clevels = {
    default_varname_mappings.HLES_AMOUNT: np.arange(-2.1, 2.2, 0.2,),
    default_varname_mappings.HLES_FREQUENCY: np.arange(-2.1, 2.2, 0.2,),
    default_varname_mappings.T_AIR_2M: np.arange(-5.5, 5.6, 1),
    default_varname_mappings.TOTAL_PREC: np.arange(-3.75, 4., 0.5),
    default_varname_mappings.LAKE_ICE_FRACTION: np.arange(-0.75, 0.8, 0.1),
    default_varname_mappings.CAO: np.arange(-2.1, 2.2, 0.2,),
    default_varname_mappings.SNOWFALL_RATE: np.arange(-3.75, 4., 0.5)
}

cc_vname_to_clevels = {
    default_varname_mappings.HLES_AMOUNT: np.arange(-2.1, 2.2, 0.2,),
    default_varname_mappings.HLES_FREQUENCY: np.arange(-2.1, 2.2, 0.2,),
    default_varname_mappings.T_AIR_2M: np.arange(-10.5, 10.6, 1),
    default_varname_mappings.TOTAL_PREC: np.arange(-3.75, 4, 0.5),
    default_varname_mappings.LAKE_ICE_FRACTION: np.arange(-0.75, 0.8, 0.1),
    default_varname_mappings.CAO: np.arange(-2.1, 2.2, 0.2,),
    default_varname_mappings.SNOWFALL_RATE: np.arange(-1.8, 2, 0.4),
}
