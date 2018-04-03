

"""
Plot the panel of cc for selected periods to all variables

for now: HLES, ice cover

"""
from collections import OrderedDict, defaultdict
from pathlib import Path

import xarray
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind

from application_properties import main_decorator
from data.robust.data_manager import DataManager
from data.robust import data_source_types
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc.cc_period import CcPeriodsInfo
from util import plot_utils
import matplotlib.pyplot as plt
from pendulum import Period
from datetime import datetime

from util.geo.mask_from_shp import get_mask
import numpy as np


def get_gl_mask(path: Path):
    """

    :param path:
    :return:
    """

    sel_file = None

    for f in path.iterdir():
        sel_file = f
        break


    with xarray.open_dataset(sel_file) as ds:
        lons, lats = [ds[k].values for k in ["lon", "lat"]]

    return get_mask(lons2d=lons, lats2d=lats, shp_path="data/shp/Great_lakes_coast_shape/gl_cst.shp") > 0.5


@main_decorator
def entry_for_cc_canesm2_gl():
    """
    for CanESM2 driven CRCM5_NEMO simulation
    """
    data_root = common_params.data_root
    label_to_datapath = OrderedDict([
        (common_params.crcm_nemo_cur_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010/merged/"),
        (common_params.crcm_nemo_fut_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100/merged/"),
    ])

    cur_st_date = datetime(1989, 1, 1)
    cur_en_date = datetime(2011, 1, 1)  # end date not inclusive

    fut_st_date = datetime(2079, 1, 1)
    fut_en_date = datetime(2101, 1, 1)  # end date not inclusive

    cur_period = Period(cur_st_date, cur_en_date)
    fut_period = Period(fut_st_date, fut_en_date)

    periods_info = CcPeriodsInfo(cur_period=cur_period, fut_period=fut_period)


    season_to_months = OrderedDict([
        ("ND", [11, 12]),
        ("JF", [1, 2]),
        ("MA", [3, 4])
    ])

    varnames = ["hles_snow", "lake_ice_fraction", "TT", "PR"]

    var_display_names = {
        "hles_snow": "HLES",
        "lake_ice_fraction": "Lake ice fraction",
        "TT": "2m air\n temperature",
        "PR": "total\nprecipitation"
    }

    plot_utils.apply_plot_params(width_cm=18, height_cm=20, font_size=8)

    the_mask = get_gl_mask(label_to_datapath[common_params.crcm_nemo_cur_label])
    vars_info = {
        "hles_snow": {
            # convert to mm/day
            "multiplier": 10,
            "display_units": "cm",
            "offset": 0,
            "vmin": -2,
            "vmax": 2,
            "accumulation": True,
            "mask": ~the_mask
        },
        "lake_ice_fraction": {
            "multiplier": 1,
            "offset": 0,
            "vmin": -0.5,
            "vmax": 0.5,
            "mask": the_mask
        },
        "TT": {
            "multiplier": 1,
            "display_units": r"${\rm ^\circ C}$",
            "offset": 0,
            "vmin": -10,
            "vmax": 10,
        },

        "PR": {
            "multiplier": 1,
            "display_units": "mm/day",
            "offset": 0,
            "vmin": -5,
            "vmax": 5,
        }

    }

    main(label_to_datapath, varnames=varnames,
         cur_label=common_params.crcm_nemo_cur_label,
         fut_label=common_params.crcm_nemo_fut_label,
         season_to_months=season_to_months,
         vname_display_names=var_display_names, periods_info=periods_info,
         vars_info=vars_info)


def calculate_change_and_pvalues(cur_data: dict, fut_data: dict, percentages=True):
    """
    :returns a dict {season: [deltas, pvals]}
    :param cur_data:
    :param fut_data:
    """
    import numpy as np
    res = OrderedDict()

    for season in cur_data:

        c = np.array([f for f in cur_data[season].values()])
        f = np.array([f for f in fut_data[season].values()])

        tval, pval = ttest_ind(c, f, axis=0, equal_var=False)

        cm = c.mean(axis=0)
        fm = f.mean(axis=0)

        if percentages:
            delta = (fm - cm) / cm * 100
        else:
            delta = fm - cm

        res[season] = [delta, pval]

    return res


def main(label_to_data_path: dict, varnames=None, season_to_months: dict=None,
         cur_label="", fut_label="",
         vname_to_mask: dict=None, vname_display_names:dict=None,
         pval_crit=0.1, periods_info: CcPeriodsInfo=None,
         vars_info: dict=None):

    """

    :param pval_crit:
    :param vars_info:
    :param label_to_data_path:
    :param varnames:
    :param season_to_months:
    :param cur_label:
    :param fut_label:
    :param vname_to_mask: - to mask everything except the region of interest
    """

    if vname_display_names is None:
        vname_display_names = {}

    varname_mapping = {v: v for v in varnames}
    level_mapping = {v: VerticalLevel(0) for v in varnames} # Does not really make a difference, since all variables are 2d

    comon_store_config = {
        DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES,
        DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: varname_mapping,
        DataManager.SP_LEVEL_MAPPING: level_mapping
    }

    cur_dm = DataManager(
        store_config=dict({DataManager.SP_BASE_FOLDER: label_to_data_path[cur_label]}, **comon_store_config)
    )

    fut_dm = DataManager(
        store_config=dict({DataManager.SP_BASE_FOLDER: label_to_data_path[fut_label]}, **comon_store_config)
    )

    # get the data and do calculations
    var_to_season_to_data = {}

    cur_start_yr, cur_end_year = periods_info.get_cur_year_limits()
    fut_start_yr, fut_end_year = periods_info.get_fut_year_limits()

    for vname in varnames:
        cur_means = cur_dm.get_seasonal_means(start_year=cur_start_yr, end_year=cur_end_year,
                                              season_to_months=season_to_months, varname_internal=vname)

        fut_means = fut_dm.get_seasonal_means(start_year=fut_start_yr, end_year=fut_end_year,
                                              season_to_months=season_to_months, varname_internal=vname)

        # convert means to the accumulators (if required)
        opts = vars_info[vname]
        if "accumulation" in opts and opts["accumulation"]:
            for seas_name, months in season_to_months.items():
                cur_means[seas_name] = {y: f * periods_info.get_numdays_for_season(y, month_list=months) for y, f in cur_means[seas_name].items()}
                fut_means[seas_name] = {y: f * periods_info.get_numdays_for_season(y, month_list=months) for y, f in fut_means[seas_name].items()}


        var_to_season_to_data[vname] = calculate_change_and_pvalues(cur_means, fut_means, percentages=False)



    # Plotting
    # panel grid dimensions
    ncols = len(season_to_months)
    nrows = len(varnames)

    gs = GridSpec(nrows, ncols)
    fig = plt.figure()

    for col, seas_name in enumerate(season_to_months):
        for row, vname in enumerate(varnames):

            ax = fig.add_subplot(gs[row, col])

            # set season titles
            if row == 0:
                ax.set_title(seas_name)

            # identify variable names
            if col == 0:
                ax.set_ylabel(vname_display_names.get(vname, vname))

            to_plot = var_to_season_to_data[vname][seas_name][0]

            vmin = None
            vmax = None
            if vars_info is not None:
                if vname in vars_info:
                    opts = vars_info[vname]
                    to_plot = to_plot * opts["multiplier"] + opts["offset"]

                    vmin = opts["vmin"]
                    vmax = opts["vmax"]


                    if "mask" in opts:
                        to_plot = np.ma.masked_where(~opts["mask"], to_plot)


            ax.set_facecolor("0.75")
            im = ax.pcolormesh(to_plot.T, cmap=cm.get_cmap("bwr", 11), vmin=vmin, vmax=vmax)
            cb = plt.colorbar(im, extend="both")
            if col < ncols - 1:
                cb.ax.set_visible(False)

    # Save the figure in file
    img_folder = common_params.img_folder
    img_folder.mkdir(exist_ok=True)

    img_file = img_folder / f"cc_{fut_label}-{cur_label}.png"

    fig.savefig(str(img_file), **common_params.image_file_options)



if __name__ == '__main__':
    entry_for_cc_canesm2_gl()