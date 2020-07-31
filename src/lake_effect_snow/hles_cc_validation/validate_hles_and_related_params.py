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
from pykdtree.kdtree import KDTree
from rpn import level_kinds
from scipy.stats import ttest_ind

from data.robust import data_source_types
from data.robust.data_manager import DataManager

from lake_effect_snow import default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import CAO, SNOWFALL_RATE, T_AIR_2M, TOTAL_PREC
from lake_effect_snow.hles_cc import common_params

from lake_effect_snow.hles_cc.common_params import var_display_names

from lake_effect_snow.data_utils import get_data as get_data_gen, all_known_variables

import sys
import logging

from lake_effect_snow.lake_effect_snowfall_entry import get_zone_around_lakes_mask
from util import plot_utils
from util.geo import lat_lon
from util.geo.mask_from_shp import get_mask
from lake_effect_snow import common_params as hles_alg_common_params


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

units = {
    default_varname_mappings.HLES_AMOUNT: "cm/day",
    default_varname_mappings.T_AIR_2M: r"${\rm ^\circ}$C",
    default_varname_mappings.TOTAL_PREC: "mm/day",
    default_varname_mappings.LAKE_ICE_FRACTION: "-",
    default_varname_mappings.HLES_FREQUENCY: "days",
    default_varname_mappings.CAO: "days",
    default_varname_mappings.SNOWFALL_RATE: "mm/day"
}

units.update({f"bias_{k}": v for k, v in units.items()})


def get_data(vname=default_varname_mappings.T_AIR_2M, season_to_months=None, beg_year=1989, end_year=2010):
    """
    Get seasonal means for each year for vname, obs and mod
    :param vname:
    """

    if season_to_months is None:
        raise ValueError("The season name to corresponding months mapping should be provided")

    obs_dir = "/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/validation_of_coupled-GL-current_CanESM2/obs"
    mod_dir = "/home/huziy/data/big1/Projects/HLES_GL_NEMO_CRCM5_CC/validation_of_coupled-GL-current_CanESM2/mod"

    data_query = {
        "obs": {
            "root_dir": obs_dir,
            "beg_year": beg_year,
            "end_year": end_year
        },
        "mod": {
            "root_dir": mod_dir,
            "beg_year": beg_year,
            "end_year": end_year
        }
    }

    return get_data_gen(vname=vname, season_to_months=season_to_months, data_query=data_query)

    #
    # the comments below are for documentation purposes
    #
    # vname_map = {
    #     default_varname_mappings.T_AIR_2M: "TT",
    #     default_varname_mappings.TOTAL_PREC: "PR",
    #     default_varname_mappings.LAKE_ICE_FRACTION: "LC",
    #     default_varname_mappings.HLES_AMOUNT: "hles_snow"
    # }
    #
    # level_map = {v: VerticalLevel(1, level_kinds.HYBRID) for v in known_variables}
    #
    # obs_multipliers = {
    #     default_varname_mappings.TOTAL_PREC: 1,  # mm/day
    #     default_varname_mappings.T_AIR_2M: 1.,
    #     default_varname_mappings.LAKE_ICE_FRACTION: 1.,
    #     default_varname_mappings.HLES_AMOUNT: 100.  # M/day -> cm/day
    # }
    #
    # mod_multipliers = obs_multipliers.copy()
    # mod_multipliers[default_varname_mappings.TOTAL_PREC] = 1.  # converted to mm/day on netcdf export
    # offset_map = defaultdict(lambda: 0)
    #
    # obs_store_config = {
    #     DataManager.SP_BASE_FOLDER: obs_dir,
    #     DataManager.SP_DATASOURCE_TYPE: data_source_types.ALL_VARS_IN_A_FOLDER_IN_NETCDF_FILES_OPEN_EACH_FILE_SEPARATELY,
    #     DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
    #     DataManager.SP_OFFSET_MAPPING: offset_map,
    #     DataManager.SP_MULTIPLIER_MAPPING: obs_multipliers,
    #     DataManager.SP_LEVEL_MAPPING: level_map
    # }
    #
    # mod_store_config = obs_store_config.copy()
    # mod_store_config[DataManager.SP_BASE_FOLDER] = mod_dir
    # mod_store_config[DataManager.SP_MULTIPLIER_MAPPING] = mod_multipliers


def __calculate_spatial_correlations(data1, data2):
    """
    datas are 2d arrays, of shape (t, x)
    :param data1:
    :param data2:
    """
    return np.corrcoef(data1.mean(axis=0), data2.mean(axis=0))[0, 1]


def calc_biases_and_pvals(v_to_data, multipliers=(1, -1)):
    """
    :param multipliers: (1, -1) yields a-b, (1,0) =>  a; for debugging purposes
    :param v_to_data: dict(variable:{"mod":<seasonal means for each year>, "obs": <seasonal mean for each year>})
    """

    v_to_bias = {}
    v_to_pvalue = {}
    v_to_corr = {}

    # extract numpy array from a dict
    def __get_np_data(a_dict):
        x = np.ma.array([a_dict[k] for k in sorted(a_dict)])
        x = np.ma.masked_where(np.isnan(x), x)
        return x

    # calculate mean biases and p-values
    for v, data in v_to_data.items():
        v_to_bias[v] = {}
        v_to_pvalue[v] = {}
        v_to_corr[v] = {}
        for season in data["mod"]:
            mod_data = __get_np_data(data["mod"][season])
            obs_data = __get_np_data(data["obs"][season])

            if v == default_varname_mappings.LAKE_ICE_FRACTION:
                mod_data = np.ma.masked_where(mod_data > 1, mod_data)
                obs_data = np.ma.masked_where(obs_data > 1, obs_data)

            annual_bias = mod_data * multipliers[0] + obs_data * multipliers[1]

            v_to_bias[v][season] = annual_bias.mean(axis=0)

            the_mask = v_to_bias[v][season].mask
            if not np.any(the_mask):
                good_points = np.ones_like(v_to_bias[v][season], dtype=np.bool)
            else:
                good_points = ~the_mask

            logger.info(good_points.shape)

            i_pos, j_pos = np.where(good_points)
            v_to_pvalue[v][season] = np.ma.masked_all_like(v_to_bias[v][season])
            pv = ttest_ind(mod_data[:, i_pos, j_pos], obs_data[:, i_pos, j_pos],
                           equal_var=False, axis=0)[1]

            r = __calculate_spatial_correlations(mod_data[:, i_pos, j_pos], obs_data[:, i_pos, j_pos])
            print(f"Correlation: var={v}, season={season}, r={r}")
            v_to_corr[v][season] = r
            v_to_pvalue[v][season][good_points] = np.ma.masked_where(np.isnan(pv), pv)

    return v_to_bias, v_to_pvalue, v_to_corr


def plot_biases(v_to_bias, v_to_pvalue, v_to_lons, v_to_lats, pval_max=0.05,
                exp_label="validation_canesm2c",
                vname_to_clevs=None, v_to_corr=None,
                var_display_names=common_params.var_display_names, img_type="pdf"):
    """
    Plot the biases on a panel, mask
    row for parameter, col for season
    :param var_display_names:
    :param pval_max:
    :param v_to_bias:
    :param v_to_pvalue:
    :param v_to_lons:
    :param v_to_lats:
    """
    logging.getLogger().setLevel(logging.INFO)
    from cartopy import crs as ccrs
    import matplotlib.pyplot as plt

    if vname_to_clevs is None:
        vname_to_clevs = common_params.bias_vname_to_clevels

    img_dir = common_params.img_folder / exp_label

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))

    var_names = list(v_to_bias.keys())
    seasons = list(v_to_bias[var_names[0]].keys())
    nrows, ncols = len(var_names), len(seasons)

    # get common map extent
    common_map_extent = None
    for var, lons in v_to_lons.items():
        if var.endswith(default_varname_mappings.HLES_AMOUNT):
            lats = v_to_lats[var]
            common_map_extent = [lons[0, 0], lons[-1, -1], lats[0, 0], lats[-1, -1]]
            logger.info(f"common_map_extent={common_map_extent}")
            break

    p = None
    clevs = None

    # box properties for the correlation value box
    bbox_properties = dict(
        boxstyle="round,pad=0.2",
        ec="k",
        fc="w",
        ls="-",
        lw=1
    )

    plot_utils.apply_plot_params(font_size=9, width_cm=50)
    plt.rcParams["axes.titlesize"] = plt.rcParams["font.size"]
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
                ax.annotate(f"{var_display_names[var]}\n", (-0.1, 0.5),
                            xycoords="axes fraction", va="center", ha="center", rotation="vertical",
                            font_properties=FontProperties(size=plt.rcParams["font.size"]))

            if ivar == 0:
                ax.set_title(season)

            lons, lats = v_to_lons[var], v_to_lats[var]

            data = v_to_bias[var][season]

            pval = v_to_pvalue[var][season]

            logger.info([pval.shape, data.shape, lons.shape, lats.shape])
            logger.info(["pval range: ", pval.min(), pval.max()])

            data = np.ma.masked_where(np.isnan(data), data)
            # to handle outside of the region of interest
            # data = np.ma.masked_where(np.abs(data) <= np.abs(data).min(), data)
            data = np.ma.masked_where((pval > pval_max), data)

            # mask outside of the HLES zone
            lake_mask = get_mask(lons, lats, shp_path=hles_alg_common_params.GL_COAST_SHP_PATH) > 0.1

            # get the KDTree for interpolation purposes
            ktree = KDTree(
                np.array(list(zip(*lat_lon.lon_lat_to_cartesian(lon=lons.flatten(), lat=lats.flatten()))))
            )

            # define the ~200km near lake zone
            near_lake_x_km_zone_mask = get_zone_around_lakes_mask(lons=lons, lats=lats, lake_mask=lake_mask,
                                                                  ktree=ktree,
                                                                  dist_km=common_params.NEAR_GL_HLES_ZONE_SIZE_KM)

            reg_of_interest = near_lake_x_km_zone_mask | lake_mask
            data = np.ma.masked_where(~reg_of_interest, data)  # actual masking happens here

            plot_field = True
            if np.all(data.mask):
                plot_field = False
                logger.info(f"all is not significant for {var} during {season}")
            else:
                logger.info(f"Plotting biases for {var} during {season}")
                assert not np.all(data.mask)

            ax.background_patch.set_facecolor("0.75")

            ax.coastlines(linewidth=0.4)
            # ax.set_xticks(np.linspace(-180, 180, 5), crs=projection)
            # ax.set_yticks(np.linspace(-90, 90, 5), crs=projection)
            # lon_formatter = LongitudeFormatter(zero_direction_label=True)
            # lat_formatter = LatitudeFormatter()
            # ax.xaxis.set_major_formatter(lon_formatter)
            # ax.yaxis.set_major_formatter(lat_formatter)

            lons[lons > 180] -= 360

            clevs = vname_to_clevs[var]
            norm = BoundaryNorm(boundaries=clevs, ncolors=len(clevs) - 1, clip=False)
            if var.startswith("bias"):
                cmap = cm.get_cmap("bwr", len(clevs) - 1)
                cmap.set_over("orange")
                cmap.set_under("cyan")
            else:
                cmap = cm.get_cmap("gist_ncar_r", len(clevs) - 1)

            assert cmap.N == len(clevs) - 1

            if plot_field:
                logger.info([var,
                             season,
                             np.sum(~data.mask) * 100. / np.product(data.shape),
                             np.sum(~data.mask)])

                p = ax.pcolormesh(lons, lats, data,
                                  transform=projection,
                                  cmap=cmap,
                                  norm=norm)
            else:
                logger.debug(f"Not plotting {var} during {season}, nothing is significant")

            line_color = "k"

            lakes_fc = "0.75"
            if var == default_varname_mappings.LAKE_ICE_FRACTION:
                lakes_fc = "none"

            ax.add_feature(common_params.LAKES_50m, facecolor="none", edgecolor=line_color, linewidth=0.5)
            ax.add_feature(common_params.COASTLINE_50m, facecolor="none", edgecolor=line_color, linewidth=0.5)
            ax.add_feature(common_params.RIVERS_50m, facecolor="none", edgecolor=line_color, linewidth=0.5)

            extent = [lons[0, 0], lons[-1, -1], lats[0, 0], lats[-1, -1]]
            logger.info(extent)
            ax.set_extent(common_map_extent, crs=projection)

            if v_to_corr is not None:
                if var in v_to_corr:
                    ax.annotate(f"$r=${v_to_corr[var][season]:.2f}", (0.05, 0.05),
                                xycoords="axes fraction", va="bottom", ha="left",
                                bbox=bbox_properties)

        # axgr.cbar_axes[0].colorbar(p, extend="both", ticks=clevs[::3])
        # axgr.cbar_axes[0].set_ylabel(units[var], fontdict=dict(size=plt.rcParams["font.size"]))

        cb_extend = "both" if var.startswith("bias") else "max"

        plt.colorbar(p, extend=cb_extend, ticks=clevs[::3], cax=axgr.cbar_axes[0])
        axgr.cbar_axes[0].set_ylabel(units[var], fontdict=dict(size=plt.rcParams["font.size"]))

    img_dir.mkdir(exist_ok=True, parents=True)
    img_file = img_dir / f"all_{exp_label}.{img_type}"
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

    known_variables = all_known_variables.copy()
    known_variables.remove(CAO)
    known_variables.remove(SNOWFALL_RATE)

    # for the 20200410 version of the validation plot
    known_variables.remove(T_AIR_2M)
    known_variables.remove(TOTAL_PREC)

    pval_max = 0.1

    v_to_data = OrderedDict()
    v_to_lons = OrderedDict()
    v_to_lats = OrderedDict()
    for v in known_variables:
        v_to_data[v], v_to_lons[v], v_to_lats[v] = get_data(v,
                                                            season_to_months=season_to_months,
                                                            beg_year=beg_year,
                                                            end_year=end_year)

    v_to_bias, v_to_pvalue, v_to_corr = calc_biases_and_pvals(v_to_data)

    v_to_obs, _, _ = calc_biases_and_pvals(v_to_data, multipliers=[0, 1])

    # mask obs as well
    for v, season_to_bias in v_to_bias.items():
        for season, bias in season_to_bias.items():
            v_to_obs[v][season] = np.ma.masked_where(bias.mask, v_to_obs[v][season])

    plot_biases(v_to_bias, v_to_pvalue, v_to_lons, v_to_lats, pval_max=pval_max,
                exp_label="validation_canesm2c_excl_tt_and_pr",
                v_to_corr=v_to_corr)


if __name__ == '__main__':
    main()
