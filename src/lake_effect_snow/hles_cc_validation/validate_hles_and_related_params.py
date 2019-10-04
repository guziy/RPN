

"""
Panel plot of biases.
1 row for parameter
2 col for season ()

"""
from collections import OrderedDict, defaultdict
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import AxesGrid
from pendulum import Period, datetime
from rpn import level_kinds
from scipy.stats import ttest_ind

from data.robust import data_source_types
from data.robust.data_manager import DataManager

from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.hles_cc import common_params

from lake_effect_snow.hles_cc.common_params import var_display_names

import sys
import logging

from util import plot_utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


known_variables = [
    default_varname_mappings.HLES_AMOUNT,
    default_varname_mappings.T_AIR_2M,
    default_varname_mappings.TOTAL_PREC,
    default_varname_mappings.LAKE_ICE_FRACTION,
]

units = {
    default_varname_mappings.HLES_AMOUNT: "cm/day",
    default_varname_mappings.T_AIR_2M: r"${\rm ^\circ}$C",
    default_varname_mappings.TOTAL_PREC: "mm/day",
    default_varname_mappings.LAKE_ICE_FRACTION: "-",

}


def get_data(vname=default_varname_mappings.T_AIR_2M, season_to_months=None, beg_year=1989, end_year=2010):
    """
    Get seasonal means for each year for vname, obs and mod
    :param vname:
    """

    if season_to_months is None:
        raise ValueError("The season name to corresponding months mapping should be provided")

    obs_dir = "/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/validation_of_coupled-GL-current_CanESM2/obs"
    mod_dir = "/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/validation_of_coupled-GL-current_CanESM2/mod"

    res = {
        "mod": None, "obs": None
    }

    if vname not in known_variables:
        raise ValueError(f"Unknown variable {vname}")

    vname_map = {
        default_varname_mappings.T_AIR_2M: "TT",
        default_varname_mappings.TOTAL_PREC: "PR",
        default_varname_mappings.LAKE_ICE_FRACTION: "LC",
        default_varname_mappings.HLES_AMOUNT: "hles_snow"
    }

    level_map = {v: VerticalLevel(1, level_kinds.HYBRID) for v in known_variables}

    obs_multipliers = {
        default_varname_mappings.TOTAL_PREC: 1,  # mm/day
        default_varname_mappings.T_AIR_2M: 1.,
        default_varname_mappings.LAKE_ICE_FRACTION: 1.,
        default_varname_mappings.HLES_AMOUNT: 100. # M/day -> cm/day
    }

    mod_multipliers = obs_multipliers.copy()
    mod_multipliers[default_varname_mappings.TOTAL_PREC] = 24 * 3600. # M/s to mm/day
    offset_map = defaultdict(lambda : 0)

    obs_store_config = {
        DataManager.SP_BASE_FOLDER: obs_dir,
        DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
        DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
        DataManager.SP_OFFSET_MAPPING: offset_map,
        DataManager.SP_MULTIPLIER_MAPPING: obs_multipliers,
        DataManager.SP_LEVEL_MAPPING: level_map
    }

    mod_store_config = obs_store_config.copy()
    mod_store_config[DataManager.SP_BASE_FOLDER] = mod_dir
    mod_store_config[DataManager.SP_MULTIPLIER_MAPPING] = mod_multipliers


    logger.debug(obs_store_config)
    logger.debug(mod_store_config)

    dm_obs = DataManager(store_config=obs_store_config)
    dm_mod = DataManager(store_config=mod_store_config)

    # calculate seasonal means
    res["mod"] = dm_mod.get_seasonal_means(start_year=beg_year, end_year=end_year, season_to_months=season_to_months,
                                           varname_internal=vname)
    res["obs"] = dm_obs.get_seasonal_means(start_year=beg_year, end_year=end_year, season_to_months=season_to_months,
                                           varname_internal=vname)

    # read some data to get coordinates
    beg_dummy = datetime(beg_year, 1, 1)
    dm_obs.read_data_for_period(Period(beg_dummy, beg_dummy.add(months=1)), varname_internal=vname)

    return res, dm_obs.lons, dm_obs.lats


def calc_biases_and_pvals(v_to_data):
    """
    :param v_to_data: dict(variable:{"mod":<seasonal means for each year>, "obs": <seasonal mean for each year>})
    """

    v_to_bias = {}
    v_to_pvalue = {}

    # extract numpy array from a dict
    def __get_np_data(a_dict):
        x = np.array([a_dict[k] for k in sorted(a_dict)])
        x = np.ma.masked_where(np.isnan(x), x)
        return x

    # calculate mean biases and p-values
    for v, data in v_to_data.items():
        v_to_bias[v] = {}
        v_to_pvalue[v] = {}
        for season in data["mod"]:
            mod_data = __get_np_data(data["mod"][season])
            obs_data = __get_np_data(data["obs"][season])

            if v == default_varname_mappings.LAKE_ICE_FRACTION:
                mod_data = np.ma.masked_where(mod_data > 1, mod_data)
                obs_data = np.ma.masked_where(obs_data > 1, obs_data)

            annual_bias = mod_data - obs_data

            v_to_bias[v][season] = annual_bias.mean(axis=0)

            good_points = ~v_to_bias[v][season].mask
            i_pos, j_pos = np.where(good_points)
            v_to_pvalue[v][season] = np.ma.masked_all_like(v_to_bias[v][season])
            v_to_pvalue[v][season][good_points] = ttest_ind(mod_data[:, i_pos, j_pos], obs_data[:, i_pos, j_pos],
                                                            equal_var=False, axis=0)[1]

    return v_to_bias, v_to_pvalue


def plot_biases(v_to_bias, v_to_pvalue, v_to_lons, v_to_lats, pval_max=0.05):
    """
    Plot the biases on a panel, mask
    row for parameter, col for season
    :param pval_max:
    :param v_to_bias:
    :param v_to_pvalue:
    :param v_to_lons:
    :param v_to_lats:
    """
    logging.getLogger().setLevel(logging.INFO)
    from cartopy import crs as ccrs
    import matplotlib.pyplot as plt

    img_dir = common_params.img_folder / "validation_canesm2c"

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))

    var_names = list(v_to_bias.keys())
    seasons = list(v_to_bias[var_names[0]].keys())
    nrows, ncols = len(var_names), len(seasons)

    # get common map extent
    common_map_extent = None
    for var, lons in v_to_lons.items():
        if var == default_varname_mappings.HLES_AMOUNT:
            lats = v_to_lats[var]
            common_map_extent = [lons[0, 0], lons[-1, -1], lats[0, 0], lats[-1, -1]]
            logger.info(f"common_map_extent={common_map_extent}")
            break

    p = None

    plot_utils.apply_plot_params(font_size=10, width_cm=40)
    fig = plt.figure()

    # create an AxesGrid for each variable
    for ivar, var in enumerate(var_names):
        axgr_layout = len(var_names) * 100 + 10 + (ivar + 1)
        axgr = AxesGrid(fig, axgr_layout, axes_class=axes_class,
                        nrows_ncols=(1, ncols),
                        axes_pad=0.1,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_pad=0.1,
                        cbar_size="6%",
                        label_mode="")  # note the empty label_mode

        for i, ax in enumerate(axgr):
            col = i % ncols
            season = seasons[col]

            if col == 0:
                ax.annotate(f"{var_display_names[var]}", (-0.1, 0.5),
                            xycoords="axes fraction", va="center", ha="center", rotation="vertical",
                            font_properties=FontProperties(size=plt.rcParams["axes.titlesize"]))

            if ivar == 0:
                ax.set_title(season)

            lons, lats = v_to_lons[var], v_to_lats[var]

            data = v_to_bias[var][season]

            pval = v_to_pvalue[var][season]

            logger.info([pval.shape, data.shape, lons.shape, lats.shape])
            logger.info(["pval range: ", pval.min(), pval.max()])

            data = np.ma.masked_where(np.isnan(data), data)
            # to handle outside of the region of interest
            # data = np.ma.masked_where(np.abs(data) < 1e-7, data)
            data = np.ma.masked_where((pval > pval_max) | pval.mask, data)

            plot_field = True
            if np.all(data.mask | np.isnan(data)):
                plot_field = False
                logger.info(f"all is not significant for {var} during {season}")
            else:
                logger.info(f"Plotting biases for {var} during {season}")

            ax.background_patch.set_facecolor("0.75")

            ax.coastlines(linewidth=0.4)
            # ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
            # ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)
            # lon_formatter = LongitudeFormatter(zero_direction_label=True)
            # lat_formatter = LatitudeFormatter()
            # ax.xaxis.set_major_formatter(lon_formatter)
            # ax.yaxis.set_major_formatter(lat_formatter)

            lons[lons > 180] -= 360

            clevs = common_params.bias_vname_to_clevels[var]
            norm = BoundaryNorm(boundaries=clevs, ncolors=len(clevs) - 1)
            cmap = cm.get_cmap("bwr", len(clevs) - 1)

            if plot_field:
                logger.debug(f"Not plotting {var} during {season}, nothing is significant")
                p = ax.contourf(lons, lats, data,
                                transform=projection,
                                cmap=cmap,
                                levels=clevs, norm=norm, extend="both")

            line_color = "k"

            ax.add_feature(common_params.LAKES_50m, facecolor="none", edgecolor=line_color, linewidth=0.5)
            ax.add_feature(common_params.COASTLINE_50m, facecolor="none", edgecolor=line_color, linewidth=0.5)
            ax.add_feature(common_params.RIVERS_50m, facecolor="none", edgecolor=line_color, linewidth=0.5)

            extent = [lons[0, 0], lons[-1, -1], lats[0, 0], lats[-1, -1]]
            logger.info(extent)
            ax.set_extent(common_map_extent, crs=projection)

        axgr.cbar_axes[0].colorbar(p)
        axgr.cbar_axes[0].set_ylabel(units[var], fontdict=dict(size=plt.rcParams["font.size"]))

    img_dir.mkdir(exist_ok=True, parents=True)
    img_file = img_dir / "all_biases.png"
    sys.stderr.write(f"Saving plots to {img_file}")
    fig.savefig(img_file, dpi=400, bbox_inches="tight")

def main():
    beg_year = 1989
    end_year = 2010
    season_to_months = OrderedDict([
        ("ND", (11, 12)),
        ("JF", (1, 2)),
        ("MA", (2, 3)),
    ])

    pval_max = 0.1

    v_to_data = OrderedDict()
    v_to_lons = OrderedDict()
    v_to_lats = OrderedDict()
    for v in known_variables:
        v_to_data[v], v_to_lons[v], v_to_lats[v] = get_data(v, season_to_months=season_to_months, beg_year=beg_year, end_year=end_year)

    v_to_bias, v_to_pvalue = calc_biases_and_pvals(v_to_data)

    plot_biases(v_to_bias, v_to_pvalue, v_to_lons, v_to_lats, pval_max=pval_max)

if __name__ == '__main__':
    main()