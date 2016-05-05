
# Some common constants for the project
from collections import OrderedDict

from rpn.rpn import RPN

dpi = 400

water_density = 1000.0

season_to_months = OrderedDict([
    ("Winter", [1, 2, 12]),
    ("Spring", [3, 4, 5]),
    ("Summer", [6, 7, 8]),
    ("Fall", [9, 10, 11])
])


def get_year_and_month(month_folder_name=""):
    ym = month_folder_name.split("_")[-1]
    return int(ym[:-2]), int(ym[-2:])


def get_nemo_lake_mask_from_rpn(fpath, vname="NEM1"):
    return RPN(fpath).get_first_record_for_name(varname=vname)