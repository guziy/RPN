
from datetime import datetime
from pathlib import Path

from pendulum import Period

cur_st_date = datetime(1989, 1, 1)
cur_en_date = datetime(2011, 1, 1)  # end date not inclusive

fut_st_date = datetime(2079, 1, 1)
fut_en_date = datetime(2101, 1, 1)  # end date not inclusive

cur_period = Period(cur_st_date, cur_en_date)
fut_period = Period(fut_st_date, fut_en_date)


data_root = Path("/Users/huziy/HLES")


img_folder = Path("hles_cc_paper")


varname_to_display_name = {
    "lake_ice_fraction": "Lake ice fraction",
    "hles_snow": "HLES"
}


crcm_nemo_cur_label = "CRCM5_NEMOc"
crcm_nemo_fut_label = "CRCM5_NEMOf"