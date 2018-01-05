import glob
from pathlib import Path

import xarray
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib import colors, cm
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap
from pendulum import Period, Pendulum
from rpn import level_kinds

from data.robust import data_source_types
from data.robust.data_manager import DataManager
from lake_effect_snow import common_params, default_varname_mappings
from lake_effect_snow.base_utils import VerticalLevel
from lake_effect_snow.default_varname_mappings import T_AIR_2M, U_WE, V_SN, vname_map_CRCM5, vname_to_offset_CRCM5, \
    vname_to_multiplier_CRCM5, vname_to_fname_prefix_CRCM5
from util import plot_utils
from util.cmap_helpers.custom_colormaps import get_with_white_added
from util.geo.misc import deg_min_sec_to_deg
from util.seasons_info import MonthPeriod

import matplotlib.pyplot as plt
import numpy as np
import pickle
import hashlib
from scipy.stats import pearsonr
from crcm5.mh_domains import default_domains




img_dir = Path("hless_stfl_link")


def get_acc_hles_and_total_snfl(ds: xarray.Dataset,
                                hles_vname="", total_snfall_vname=""):
    return [ds[vn].sum(dim="t").values for vn in [hles_vname, total_snfall_vname]]


def plot_avg_snfall_maps(b: Basemap, xx, yy, hles_snfall, total_snfall,
                         cmap=None, bnorm: BoundaryNorm=None, label=""):
    fig = plt.figure()

    gs = GridSpec(1, 3, wspace=0.01)

    h = hles_snfall.mean(axis=0)
    t = total_snfall.mean(axis=0)



    axes_list = []

    # hles
    ax = fig.add_subplot(gs[0, 0])
    cs = b.contourf(xx, yy, h, bnorm.boundaries, cmap=cmap, norm=bnorm, extend="max")
    ax.set_title("HLES (cm)")
    cb = b.colorbar(cs, location="bottom")
    cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=45)
    axes_list.append(ax)


    # total
    ax = fig.add_subplot(gs[0, 1])
    cs = b.contourf(xx, yy, t, bnorm.boundaries, cmap=cmap, norm=bnorm, extend="max")
    ax.set_title("Snowfall (cm)")
    cb = b.colorbar(cs, location="bottom")
    cb.ax.set_visible(False)
    axes_list.append(ax)


    # hles percentage
    ax = fig.add_subplot(gs[0, 2])
    clevs = np.arange(0, 100, 10)
    cs = b.contourf(xx, yy, h / t * 100,
                    levels=clevs,
                    cmap=get_with_white_added("gist_ncar_r", white_end=0.1,
                                              ncolors_out=len(clevs) - 1))

    ax.set_title("HLES fraction (%)")
    cb = b.colorbar(cs, location="bottom")
    axes_list.append(ax)


    # plot coastlines
    for ax in axes_list:
        b.drawcoastlines(ax=ax, linewidth=0.5)


    fig.savefig(str(img_dir / f"mean_hles_total_snfall_{label}.png"),
                bbox_inches="tight", dpi=250)
    plt.close(fig)


def plot_area_avg_snfall(hles_snfall, total_snfall, hles_period_list, label="",
                         hles_crit_fraction=0.1):
    hles_region = (hles_snfall.mean(axis=0) / total_snfall.mean(axis=0) >= hles_crit_fraction)

    i_arr, j_arr = np.where(hles_region)

    hles_s = hles_snfall[:, i_arr, j_arr].mean(axis=1)
    total_s = total_snfall[:, i_arr, j_arr].mean(axis=1)
    ylist = [p.end.year for p in hles_period_list]


    gs = GridSpec(2, 1, hspace=0.1)

    fig = plt.figure()

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ylist, hles_s, lw=2, label="HLES", marker="s")
    ax.plot(ylist, total_s, lw=2, label="Total", marker="s")
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ylist, hles_s / total_s * 100, lw=2, label="HLES/Total (%)", marker="s")
    ax.grid(True)
    ax.legend()


    # rotate ticklabels
    # for ax in fig.axes:
    #     for tick_label in ax.get_xticklabels():
    #         tick_label.set_rotation(45)


    fig.savefig(str(img_dir / f"{label}_area_avg_ts.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_area_avg_snfall_and_stfl(hles_snfall, total_snfall, hles_period_list, stfl_series, label="",
                         hles_crit_fraction=0.1, stfl_months_of_interest=tuple(range(1, 6))):
    hles_region = (hles_snfall.mean(axis=0) / total_snfall.mean(axis=0) >= hles_crit_fraction)

    i_arr, j_arr = np.where(hles_region)

    hles_s = hles_snfall[:, i_arr, j_arr].mean(axis=1)
    total_s = total_snfall[:, i_arr, j_arr].mean(axis=1)
    ylist = [p.end.year for p in hles_period_list]


    stfl = stfl_series.squeeze().to_series()
    stfl.index = stfl_series.coords["t"].values

    stfl = stfl[stfl.index.month.isin(stfl_months_of_interest)]
    stfl = stfl.groupby(stfl.index.year).max()

    stfl = stfl[[p.end.year for p in hles_period_list]].values



    gs = GridSpec(2, 1, hspace=0.1)

    fig = plt.figure()

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ylist, hles_s, lw=2, label="HLES", marker="s")
    ax.plot(ylist, total_s, lw=2, label="Total", marker="s")
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ylist, stfl, lw=2, label="Streamflow (m$^3$/s)", marker="s")
    ax.grid(True)
    ax.legend()

    # rotate ticklabels
    # for ax in fig.axes:
    #     for tick_label in ax.get_xticklabels():
    #         tick_label.set_rotation(45)


    fig.savefig(str(img_dir / f"{label}_area_avg_ts_with_stfl.png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def __add_subplot(fig, gs: GridSpec, row: int, col: int, ax_list=None):
    ax = fig.add_subplot(gs[row, col])
    ax_list.append(ax)
    return ax


def plot_correlation_maps_with_stfl(b: Basemap, xx, yy,
                                    hles_snfall, total_snfall, period_list, stfl_series: xarray.DataArray,
                                    label="", stfl_months_of_interest=tuple(range(1, 6)),
                                    hles_crit_fraction=0.1):

    hles_region = (hles_snfall.mean(axis=0) / total_snfall.mean(axis=0) >= hles_crit_fraction)
    i_arr, j_arr = np.where(hles_region)


    # get the coordinates of the streamflow point
    stfl_lon, stfl_lat = stfl_series.coords["lon"].values[0, 0], stfl_series.coords["lat"].values[0, 0]
    stfl_x, stfl_y = b(stfl_lon, stfl_lat)


    stfl = stfl_series.squeeze().to_series()
    stfl.index = stfl_series.coords["t"].values

    stfl = stfl[stfl.index.month.isin(stfl_months_of_interest)]
    stfl = stfl.groupby(stfl.index.year).max()

    stfl = stfl[[p.end.year for p in period_list]].values

    assert len(stfl) == len(period_list)
    assert len(stfl) == total_snfall.shape[0]


    hles_corr = np.ma.masked_all_like(xx)
    pval_hles_corr = np.ma.masked_all_like(xx)

    total_corr = np.ma.masked_all_like(yy)
    pval_total_corr = np.ma.masked_all_like(yy)

    for i, j in zip(i_arr, j_arr):
        hles_corr[i, j], pval_hles_corr[i, j] = pearsonr(hles_snfall[:, i, j], stfl)
        total_corr[i, j], pval_total_corr[i, j] = pearsonr(total_snfall[:, i, j], stfl)


    # plot correlations and pvalues
    corr_levs = list(np.arange(-1, 0, 0.1)) + list(np.arange(0.1, 1.1, 0.1))

    corr_cmap = get_with_white_added("coolwarm",
                                     white_start=0.45, white_end=0.55,
                                     ncolors_out=len(corr_levs) - 1)
    corr_norm = BoundaryNorm(corr_levs, len(corr_levs) - 1)


    pval_levs = [0, 0.01, 0.05, 0.1, 0.2]
    pval_cmap = cm.get_cmap("tab10", len(pval_levs) - 1)
    pval_norm = BoundaryNorm(pval_levs, len(pval_levs) - 1)



    gs = GridSpec(2, 2, wspace=0.05, hspace=0.)

    fig = plt.figure()

    axes_list = []
    ax = __add_subplot(fig, gs, 0, 0, ax_list=axes_list)
    cs = b.pcolormesh(xx, yy, total_corr, cmap=corr_cmap, norm=corr_norm, ax=ax)
    ax.set_title("Corr.")
    ax.set_ylabel("Total snowfall")
    cb = b.colorbar(cs, location="bottom")
    cb.ax.set_visible(False)


    ax = __add_subplot(fig, gs, 0, 1, ax_list=axes_list)
    cs = b.pcolormesh(xx, yy, pval_total_corr, cmap=pval_cmap, norm=pval_norm, ax=ax)
    ax.set_title("p-value")
    cb = b.colorbar(cs, location="bottom", extend="max")
    cb.ax.set_visible(False)


    # HLES
    ax = __add_subplot(fig, gs, 1, 0, ax_list=axes_list)
    cs = b.pcolormesh(xx, yy, hles_corr, cmap=corr_cmap, norm=corr_norm, ax=ax)
    ax.set_ylabel("HLES")
    b.colorbar(cs, location="bottom")

    ax = __add_subplot(fig, gs, 1, 1, ax_list=axes_list)
    cs = b.pcolormesh(xx, yy, pval_hles_corr, cmap=pval_cmap, norm=pval_norm, ax=ax)
    b.colorbar(cs, location="bottom", extend="max")



    # common adjustments
    for ax in axes_list:
        b.drawcoastlines(linewidth=0.5, ax=ax)
        b.scatter(stfl_x, stfl_y, 200, marker="*", zorder=10, color="m", ax=ax)

        # draw basin boundaries
        # add basin patches
        b.readshapefile(default_domains.GL_BASINS_FROM_MUSIC_BILJANA_PATH[:-4],
                        "basin",
                        linewidth=2,
                        color="m", zorder=12)

        patches = []

        for info, shape in zip(b.basin_info, b.basin):
            if info["LAKEBASIN"].lower().endswith("lawrence"): # Skip ottawa river basin
                continue

            # convert to the [0, 360) range
            sh = np.array(shape)
            x = sh[:, 0]
            x[x < 0] += 360
            patches.append(Polygon(sh, True))

        ax.add_collection(PatchCollection(patches, facecolor='none', edgecolor='b', linewidths=0.2, zorder=2))

    fig.savefig(str(img_dir / f"correlation_maps_{label}.png"),
                dpi=250, bbox_inches="tight")

    plt.close(fig)




def plot_stfl_for_min_and_max_hles_years(b: Basemap, xx, yy,
                                         hles_snfall, total_snfall, stfl_series, label=""):
    pass




def get_streamflow_at(lon=-100., lat=50., data_source_base_dir="",
                      period=None, varname=default_varname_mappings.STREAMFLOW):


    """
    Uses caching
    :param lon:
    :param lat:
    :param data_source_base_dir:
    :param period:
    :param varname:
    :return:
    """
    cache_dir = Path("point_data_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    bd_sha = hashlib.sha224(data_source_base_dir.encode()).hexdigest()

    cache_file = cache_dir / f"{varname}_lon{lon}_lat{lat}_{period.start}-{period.end}_{bd_sha}.bin"


    if cache_file.exists():
        return pickle.load(cache_file.open("rb"))

    vname_to_level_erai = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
    }

    vname_map = {}
    vname_map.update(vname_map_CRCM5)

    store_config = {
            DataManager.SP_BASE_FOLDER: data_source_base_dir,
            DataManager.SP_DATASOURCE_TYPE: data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT,
            DataManager.SP_INTERNAL_TO_INPUT_VNAME_MAPPING: vname_map,
            DataManager.SP_LEVEL_MAPPING: vname_to_level_erai,
            DataManager.SP_OFFSET_MAPPING: vname_to_offset_CRCM5,
            DataManager.SP_MULTIPLIER_MAPPING: vname_to_multiplier_CRCM5,
            DataManager.SP_VARNAME_TO_FILENAME_PREFIX_MAPPING: vname_to_fname_prefix_CRCM5,
    }

    dm = DataManager(store_config=store_config)


    lons_ = np.asarray([lon])
    lats_ = np.asarray([lat])

    data = dm.read_data_for_period_and_interpolate(
        period=period, varname_internal=varname,
        lons_target=lons_, lats_target=lats_
    )

    pickle.dump(data, cache_file.open("wb"))
    return data


def main():
    hless_data_path = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2015_monthly"



    print(f"HLES data source: {hless_data_path}")

    # create a directory for images
    img_dir.mkdir(parents=True, exist_ok=True)

    month_period = MonthPeriod(start_month=11, nmonths=3)

    hles_vname = "hles_snow"
    total_snfall_vname = "total_snowfall"

    start_year = 1980
    end_year = 2009

    stfl_data_source_base_dir = "/snow3/huziy/NEI/GL/erai0.75deg_driven/GL_with_NEMO_dtN_1h_and_30min/Samples/"
    stfl_period = Period(
        Pendulum(start_year, 1, 1), Pendulum(end_year + 1, 1, 1)
    )



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
    total_snfall = np.array(total_snfall)

    #convert to cm
    hles_snfall *= 100
    total_snfall *= 100


    # upstream of Cornwall
    stfl_lon = 284.64685
    stfl_lat = 44.873371

    stfl_data = get_streamflow_at(stfl_lon, stfl_lat,
                                  data_source_base_dir=stfl_data_source_base_dir,
                                  period=stfl_period,
                                  varname=default_varname_mappings.STREAMFLOW)


    # stfl_data.plot()
    # plt.show()






    # do the plotting
    plot_utils.apply_plot_params(font_size=10)

    snow_clevs = np.array(common_params.clevs_lkeff_snowfall) * 1.25
    # cmap, bn = colors.from_levels_and_colors(snow_clevs,
    #                                          ["white", "indigo", "blue", "dodgerblue", "aqua", "lime", "yellow", "gold",
    #                                           "orange", "red"][:len(snow_clevs)],
    #                                          extend="max")

    cmap = LinearSegmentedColormap.from_list(
        "mycmap", common_params.lkeff_snowfall_colors,
        N=len(common_params.lkeff_snowfall_colors)
    )
    bn = BoundaryNorm(snow_clevs, len(snow_clevs) - 1)



    b = Basemap(
        llcrnrlon=lons[0, 0], llcrnrlat=lats[0, 0],
        urcrnrlon=lons[-1, -1], urcrnrlat=lats[-1, -1],
        resolution="i", area_thresh=1000
    )

    xx, yy = b(lons, lats)
    # b.drawcoastlines()






    plot_avg_snfall_maps(b, xx, yy, hles_snfall, total_snfall,
                         cmap=cmap,
                         bnorm=bn,
                         label=Path(hless_data_path).name)


    # plot area avg timeseries
    plot_area_avg_snfall(
        hles_snfall, total_snfall, hles_period_list=period_list,
        label=Path(hless_data_path).name
    )

    # plot correlation maps
    plot_correlation_maps_with_stfl(b, xx, yy, hles_snfall, total_snfall, period_list=period_list,
                                    stfl_series=stfl_data, label=Path(hless_data_path).name,
                                    stfl_months_of_interest=tuple(range(1, 5)))

    #
    plot_area_avg_snfall_and_stfl(
        hles_snfall, total_snfall, stfl_series=stfl_data, hles_period_list=period_list,
        label=Path(hless_data_path).name, stfl_months_of_interest=tuple(range(1, 5))
    )


if __name__ == '__main__':
    main()
