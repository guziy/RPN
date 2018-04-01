
from pathlib import Path




data_root = Path("/Users/huziy/HLES")


img_folder = Path("hles_cc_paper")


varname_to_display_name = {
    "lake_ice_fraction": "Lake ice fraction",
    "hles_snow": "HLES"
}


crcm_nemo_cur_label = "CRCM5_NEMOc"
crcm_nemo_fut_label = "CRCM5_NEMOf"


image_file_options = dict(
    bbox_inches="tight", dpi=300
)