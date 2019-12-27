import glob
from collections import OrderedDict
from pathlib import Path

import xarray
from eofs.standard import Eof
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans, Basemap

from application_properties import main_decorator
from lake_effect_snow import common_params
from lake_effect_snow.analyse_stfl_hless_link.total_and_hless_snfall import get_acc_hles_and_total_snfl
from util import plot_utils
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from matplotlib import colors

import numpy as np

import pandas as pd

from util.seasons_info import MonthPeriod

img_dir = Path("hles_results_analysis_using_monthly_files")


@main_decorator
def main():
    clevs_lkeff_snowfalldays = [0, 0.1, 0.8, 1.6, 2.4, 3.2, 4.0, 5]
    clevs_lkeff_snowfall = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10]
    mycolors = ["white", "indigo", "blue", "dodgerblue", "aqua", "lime", "yellow", "gold",
                                                     "orange", "orangered", "red", "firebrick", ]


    start_year = 1980
    end_year = 2009


    label_to_hles_dir = OrderedDict(
        [
         ("Obs", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_icefix_1980-2009")),
         ("CRCM5_NEMO", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2015_monthly")),
         # ("CRCM5_HL", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_Hostetler_1980-2009")),
         # ("CRCM5_NEMO_TT_PR", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_based_on_TT_PR_1980-2009"))
        ]
    )


    label_to_line_style = {
        "Obs": "k.-",
        "CRCM5_NEMO": "r",
        "CRCM5_HL": "b",
        "CRCM5_NEMO_TT_PR": "g"
    }




    clevs_hles = clevs_lkeff_snowfall
    vname_hles = "snow_fall"
    units_hles = "cm"

    clevs_hles_days = clevs_lkeff_snowfalldays
    vname_hles_days = "lkeff_snowfall_days"
    units_hles_days = "days"

    vname = vname_hles
    if vname == vname_hles_days:
        clevs = clevs_hles_days
        units = units_hles_days
    else:
        clevs = clevs_hles
        units = units_hles


    cmap, bn = colors.from_levels_and_colors(clevs, mycolors[:len(clevs) - 1])
    cmap.set_over(mycolors[len(clevs) - 2])


    label_to_y_to_snfl = {}
    label_to_pc = {}

    label_to_eof = OrderedDict()
    label_to_varfraction = OrderedDict()

    mask = None

    plot_utils.apply_plot_params(font_size=12)

    # create a directory for images
    img_dir.mkdir(parents=True, exist_ok=True)


    label_to_hles_data = OrderedDict()

    for label, hless_data_path in label_to_hles_dir.items():

        # hless_data_path = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2015_monthly"

        print(f"HLES data source: {hless_data_path}")


        month_period = MonthPeriod(start_month=11, nmonths=3)

        hles_vname = "hles_snow"
        total_snfall_vname = "total_snowfall"

        hles_snfall = []
        total_snfall = []
        period_list = []

        lons, lats = None, None
        for p in month_period.get_season_periods(start_year=start_year, end_year=end_year):
            print(p.start, p.end)
            flist = []
            for start in p.range("months"):
                y = start.year
                m = start.month
                a_file = glob.glob(f"{hless_data_path}/*{y}-{y}_m{m:02d}-{m:02d}_daily.nc")[0]
                flist.append(a_file)

            ds = xarray.open_mfdataset(flist, data_vars="minimal")

            if lons is None:
                lons, lats = [ds[k][:].values for k in ["lon", "lat"]]

            hles1, total1 = get_acc_hles_and_total_snfl(ds,
                                                        hles_vname=hles_vname,
                                                        total_snfall_vname=total_snfall_vname)

            hles_snfall.append(hles1)
            total_snfall.append(total1)
            period_list.append(p)

        # convert annual mean
        hles_snfall = np.array(hles_snfall)
        total_snfall = np.array(total_snfall)  # <-- not using this one

        label_to_hles_data[label] = hles_snfall


