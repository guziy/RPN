import cartopy
from collections import OrderedDict
from datetime import datetime

from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pendulum import Period

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler.main_for_lake_effect_snow import get_mask_of_points_near_lakes
from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc.cc_period import CcPeriodsInfo
from lake_effect_snow.hles_cc.plot_cc_2d_all_variables_for_all_periods import get_gl_mask
from util import plot_utils

import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt


@main_decorator
def entry_for_cc_canesm2_gl():
    """
    for CanESM2 driven CRCM5_NEMO simulation
    """
    data_root = common_params.data_root
    label_to_datapath = OrderedDict([
        (common_params.crcm_nemo_cur_label,
         data_root / "cur/hles"),
        (common_params.crcm_nemo_fut_label,
         data_root / "fut/hles"),
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

    var_pairs = [("hles_snow", "TT"), ("hles_snow", "PR"), ("hles_snow", "lake_ice_fraction")]

    var_display_names = {
        "hles_snow": "HLES",
        "lake_ice_fraction": "Mean Lake ice \nfraction",
        "TT": "2m air\n temperature",
        "PR": "total\nprecipitation"
    }

    plot_utils.apply_plot_params(width_cm=25, height_cm=25, font_size=8)

    gl_mask = get_gl_mask(label_to_datapath[common_params.crcm_nemo_cur_label])
    hles_region_mask = get_mask_of_points_near_lakes(gl_mask, npoints_radius=20)

    main(label_to_data_path=label_to_datapath,
         var_pairs=var_pairs, periods_info=periods_info,
         vname_display_names=var_display_names, season_to_months=season_to_months,
         hles_region_mask=hles_region_mask, lakes_mask=gl_mask)


def calculate_correlations_and_pvalues(var_pairs,
                                       label_to_vname_to_season_to_yearlydata: dict,
                                       season_to_months: dict,
                                       region_of_interest_mask, lakes_mask=None, lats=None) -> dict:
    """

    :param var_pairs:
    :param label_to_vname_to_season_to_yearlydata:
    :param lats needed for weighting of eof solver
    :return: {(vname1, vname2): {label: {season: [corr, pvalue]}}}}
    """
    res = {}
    for pair in var_pairs:
        pair = tuple(pair)

        res[pair] = {}

        for label in label_to_vname_to_season_to_yearlydata:

            res[pair][label] = {}
            for season in season_to_months:

                years_sorted = sorted(label_to_vname_to_season_to_yearlydata[label][pair[0]][season])

                v1_dict, v2_dict = [label_to_vname_to_season_to_yearlydata[label][pair[vi]][season] for vi in range(2)]
                v1 = np.array([v1_dict[y] for y in years_sorted])
                v2 = np.array([v2_dict[y] for y in years_sorted])

                r = np.zeros(v1.shape[1:]).flatten()
                p = np.ones_like(r).flatten()

                v1 = v1.reshape((v1.shape[0], -1))
                v2 = v2.reshape((v2.shape[0], -1))

                # for hles and ice fraction get the eof of the ice and correlate
                if pair == ("hles_snow", "lake_ice_fraction"):
                    # assume that v2 is the lake_ice_fraction
                    v_lake_ice = v2

                    positions_hles_region = np.where(region_of_interest_mask.flatten())[0]
                    positions_lakes = np.where(lakes_mask.flatten())[0]

                    v_lake_ice = v_lake_ice[:, positions_lakes]
                    # calculate anomalies
                    area_avg_lake_ice = (v_lake_ice - v_lake_ice.mean(axis=0)[np.newaxis, :]).mean(axis=1)

                    # print(positions)
                    for i in positions_hles_region:
                        r[i], p[i] = pearsonr(v1[:, i], area_avg_lake_ice)

                else:

                    positions = np.where(region_of_interest_mask.flatten())

                    # print(positions)
                    for i in positions[0]:
                        r[i], p[i] = pearsonr(v1[:, i], v2[:, i])

                r.shape = region_of_interest_mask.shape
                p.shape = region_of_interest_mask.shape

                r = np.ma.masked_where(~region_of_interest_mask, r)
                p = np.ma.masked_where(~region_of_interest_mask, p)

                res[pair][label][season] = [r, p]

    return res


def main(label_to_data_path: dict, var_pairs: list,
         periods_info: CcPeriodsInfo,
         vname_display_names=None,
         season_to_months: dict = None,
         cur_label=common_params.crcm_nemo_cur_label,
         fut_label=common_params.crcm_nemo_fut_label,
         hles_region_mask=None, lakes_mask=None):
    # get a flat list of all the required variable names (unique)
    varnames = []
    for vpair in var_pairs:
        for v in vpair:
            if v not in varnames:
                varnames.append(v)

    print(f"Considering {varnames}, based on {var_pairs}")

    if vname_display_names is None:
        vname_display_names = {}

    varname_mapping = {v: v for v in varnames}
    level_mapping = {v: VerticalLevel(0) for v in
                     varnames}  # Does not really make a difference, since all variables are 2d

    common_store_config = {
        DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES,
        DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: varname_mapping,
        DataManager.SP_LEVEL_MAPPING: level_mapping
    }

    cur_dm = DataManager(
        store_config=dict({DataManager.SP_BASE_FOLDER: label_to_data_path[cur_label]}, **common_store_config)
    )

    fut_dm = DataManager(
        store_config=dict({DataManager.SP_BASE_FOLDER: label_to_data_path[fut_label]}, **common_store_config)
    )

    # get the data and do calculations
    label_to_vname_to_season_to_data = {}

    cur_start_yr, cur_end_year = periods_info.get_cur_year_limits()
    fut_start_yr, fut_end_year = periods_info.get_fut_year_limits()

    # load coordinates in memory
    cur_dm.read_data_for_period(Period(datetime(cur_start_yr, 1, 1), datetime(cur_start_yr, 1, 2)),
                                varname_internal=varnames[0])

    label_to_vname_to_season_to_data = {
        cur_label: {}, fut_label: {}
    }

    for vname in varnames:
        cur_means = cur_dm.get_seasonal_means(start_year=cur_start_yr, end_year=cur_end_year,
                                              season_to_months=season_to_months, varname_internal=vname)

        fut_means = fut_dm.get_seasonal_means(start_year=fut_start_yr, end_year=fut_end_year,
                                              season_to_months=season_to_months, varname_internal=vname)

        label_to_vname_to_season_to_data[cur_label][vname] = cur_means
        label_to_vname_to_season_to_data[fut_label][vname] = fut_means

    if hles_region_mask is None:
        data_field = label_to_vname_to_season_to_data[common_params.crcm_nemo_cur_label][list(season_to_months.keys())[0]]
        hles_region_mask = np.ones_like(data_field)

    correlation_data = calculate_correlations_and_pvalues(var_pairs, label_to_vname_to_season_to_data,
                                                          season_to_months=season_to_months,
                                                          region_of_interest_mask=hles_region_mask,
                                                          lats=cur_dm.lats, lakes_mask=lakes_mask)

    # Calculate mean seasonal temperature
    label_to_season_to_tt_mean = {}
    for label, vname_to_season_to_data in label_to_vname_to_season_to_data.items():
        label_to_season_to_tt_mean[label] = {}
        for season, yearly_data in vname_to_season_to_data["TT"].items():
            label_to_season_to_tt_mean[label][season] = np.mean([f for f in yearly_data.values()], axis=0)

    # do the plotting
    fig = plt.figure()

    ncols = len(season_to_months)
    nrows = len(var_pairs) * len(label_to_vname_to_season_to_data)

    gs = GridSpec(nrows, ncols, wspace=0, hspace=0)

    for col, season in enumerate(season_to_months):
        row = 0

        for vpair in var_pairs:
            for label in sorted(label_to_vname_to_season_to_data):
                ax = fig.add_subplot(gs[row, col], projection=cartopy.crs.PlateCarree())

                r, pv = correlation_data[vpair][label][season]

                r[np.isnan(r)] = 0
                r = np.ma.masked_where(~hles_region_mask, r)
                ax.set_facecolor("0.75")

                # hide the ticks
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())

                im = ax.pcolormesh(cur_dm.lons, cur_dm.lats, r, cmap=cm.get_cmap("bwr", 11), vmin=-1, vmax=1)

                # add 0 deg line
                cs = ax.contour(cur_dm.lons, cur_dm.lats, label_to_season_to_tt_mean[label][season], levels=[0,],
                                linewidths=1, colors="k")
                ax.set_extent([cur_dm.lons[0, 0], cur_dm.lons[-1, -1], cur_dm.lats[0, 0], cur_dm.lats[-1, -1]])

                ax.background_patch.set_facecolor("0.75")

                if row == 0:
                    # ax.set_title(season + f", {vname_display_names[vpair[0]]}")
                    ax.text(0.5, 1.05, season, transform=ax.transAxes,
                            va="bottom", ha="center", multialignment="center")

                if col == 0:
                    # ax.set_ylabel(f"HLES\nvs {vname_display_names[vpair[1]]}\n{label}")
                    ax.text(-0.05, 0.5, f"HLES\nvs {vname_display_names[vpair[1]]}\n{label}",
                            va="center", ha="right",
                            multialignment="center",
                            rotation=90,
                            transform=ax.transAxes)

                divider = make_axes_locatable(ax)
                ax_cb = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
                fig.add_axes(ax_cb)
                cb = plt.colorbar(im, extend="both", cax=ax_cb)

                if row < nrows - 1 or col < ncols - 1:
                    cb.ax.set_visible(False)

                row += 1

    img_dir = common_params.img_folder
    img_dir.mkdir(exist_ok=True)

    img_file = img_dir / "hles_tt_pr_correlation_fields_cur_and_fut_mean_ice_fraction.png"
    fig.savefig(str(img_file), **common_params.image_file_options)


if __name__ == '__main__':
    entry_for_cc_canesm2_gl()
