# plot monthly histograms for the CanESM2-driven simulations future vs current
from collections import OrderedDict

from datetime import datetime

import xarray
from matplotlib.dates import MonthLocator, num2date, date2num
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.spatial import KDTree

from application_properties import main_decorator
from crcm5.nemo_vs_hostetler.main_for_lake_effect_snow import get_mask_of_points_near_lakes
from lake_effect_snow.hles_cc import common_params
from lake_effect_snow.hles_cc.plot_cc_2d_all_variables_for_all_periods import get_gl_mask
from lake_effect_snow.hles_cc.plot_domain_from_ncfile_using_cartopy import plot_domain_and_interest_region
from lake_effect_snow.plot_monthly_histograms import get_monthly_accumulations_area_avg_from_merged
from util import plot_utils
import matplotlib.pyplot as plt
import numpy as np
from rpn.domains import lat_lon

import logging

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def get_hles_amount_distribution_from_merged(data_file, varname="hles_snow", bin_edges=None,
                                             region_of_interest_mask=None, selected_months=None):
    """
    Assumes that the input file contains data in cm/day
    :param selected_months:
    :param region_of_interest_mask:
    :param data_file:
    :param varname:
    """

    if selected_months is None:
        selected_months = list(range(1, 13))

    i_arr, j_arr = np.where(region_of_interest_mask)

    with xarray.open_dataset(data_file) as ds:

        logger.debug(ds["t.month"].isin(selected_months))

        # select time first
        logger.debug(ds[varname])

        logger.debug(["arr.shape=", ds[varname].shape])
        arr = ds[varname][ds["t.month"].isin(selected_months)].data[:, i_arr, j_arr]

        logger.debug(["arr.shape=", arr.shape])

        # ignore hles left of the leftmost edge
        arr = arr[arr >= bin_edges[0]]

        logger.debug(["arr.shape=", arr.shape])

        return np.histogram(arr, bins=bin_edges)[0]



@main_decorator
def main(varname=""):
    plot_utils.apply_plot_params(width_cm=22, height_cm=5, font_size=8)
    # series = get_monthly_accumulations_area_avg(data_dir="/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009",
    #                                             varname=varname)

    # series = get_monthly_accumulations_area_avg(data_dir="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2009_monthly",
    #                                             varname=varname)

    # series = get_monthly_accumulations_area_avg(data_dir="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_HL_1980-2009_monthly",
    #                                             varname=varname)

    hles_bin_edges = np.arange(0.1, 0.34, 0.02)

    # selected_months = [10, 11, 12, 1, 2, 3, 4, 5]
    selected_seasons = OrderedDict([
        ("ND", [11, 12]), ("JF", [1, 2]), ("MA", [3, 4]), ("NDJFMA", [11, 12, 1, 2, 3, 4])
    ])

    data_root = common_params.data_root

    label_to_datapath = OrderedDict([
        # ("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_1980-2009"),
        # ("Obs", "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_daily_Obs_monthly_icefix_1980-2009"),
        # (common_params.crcm_nemo_cur_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-2010_1989-2010" / "merged"),
        # (common_params.crcm_nemo_fut_label, data_root / "lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2100_2079-2100" / "merged"),
        (common_params.crcm_nemo_cur_label,
         data_root / "lake_effect_analysis_CRCM5_NEMO_fix_CanESM2_RCP85_1989-2010_monthly_1989-2010" / "merged"),
        (common_params.crcm_nemo_fut_label,
         data_root / "lake_effect_analysis_CRCM5_NEMO_fix_CanESM2_RCP85_2079-2100_monthly_2079-2100" / "merged"),
    ])

    # longutudes and latitudes of the focus region around the Great Lakes (we define it, mostly for performance
    # issues and to eliminate regions with 0 hles that still are in the 200 km HLES zone)
    focus_region_lonlat_nc_file = data_root / "lon_lat.nc"

    label_to_series = OrderedDict()
    label_to_color = {
        common_params.crcm_nemo_cur_label: "skyblue",
        common_params.crcm_nemo_fut_label: "salmon"

    }

    gl_mask = get_gl_mask(label_to_datapath[common_params.crcm_nemo_cur_label])
    hles_region_mask = get_mask_of_points_near_lakes(gl_mask, npoints_radius=20)

    # select a file from the directory
    sel_file = None
    for f in label_to_datapath[common_params.crcm_nemo_cur_label].iterdir():
        if f.is_file():
            sel_file = f
            break

    assert sel_file is not None, f"Could not find any files in {label_to_datapath[common_params.crcm_nemo_cur_label]}"

    # Take into account the focus region
    with xarray.open_dataset(sel_file) as ds:
        hles_region_mask_lons, hles_region_mask_lats = [ds[k].values for k in ["lon", "lat"]]

        with xarray.open_dataset(focus_region_lonlat_nc_file) as ds_focus:
            focus_lons, focus_lats = [ds_focus[k].values for k in ["lon", "lat"]]

        coords_src = lat_lon.lon_lat_to_cartesian(hles_region_mask_lons.flatten(), hles_region_mask_lats.flatten())
        coords_dst = lat_lon.lon_lat_to_cartesian(focus_lons.flatten(), focus_lats.flatten())

        ktree = KDTree(list(zip(*coords_src)))

        dists, inds = ktree.query(list(zip(*coords_dst)), k=1)

        focus_mask = hles_region_mask.flatten()
        focus_mask[...] = False
        focus_mask[inds] = True
        focus_mask.shape = hles_region_mask.shape

    for seas_name, selected_months in selected_seasons.items():
        # read and calculate
        for label, datapath in label_to_datapath.items():
            hles_file = None

            # select hles file in the folder
            for f in datapath.iterdir():
                if f.name.endswith("_daily.nc"):
                    hles_file = f
                    break

            assert hles_file is not None, f"Could not find any HLES files in {datapath}"

            series = get_hles_amount_distribution_from_merged(data_file=hles_file, varname=varname,
                                                              region_of_interest_mask=hles_region_mask & focus_mask,
                                                              selected_months=selected_months, bin_edges=hles_bin_edges)
            label_to_series[label] = series

        #  plotting
        gs = GridSpec(1, 1, wspace=0.05)

        fig = plt.figure()
        ax = fig.add_subplot(gs[0, 0])

        # calculate bar widths
        widths = np.diff(hles_bin_edges)

        label_to_handle = OrderedDict()

        for i, (label, series) in enumerate(label_to_series.items()):
            values = series.values if hasattr(series, "values") else series

            # values = values / values.sum() * 100

            logger.debug([label, values])
            logger.debug(f"sum(values) = {sum(values)}")

            # h = ax.bar(hles_bin_edges[:-1] + i * widths / len(label_to_series), values, width=widths / len(label_to_series),
            #            align="edge", linewidth=0.5,
            #            edgecolor="k",
            #            facecolor=label_to_color[label], label=label, zorder=10)

            h = ax.plot(hles_bin_edges[:-1], values, color=label_to_color[label], marker="o", label=label,
                        markersize=2.5)
            # label_to_handle[label] = h

        ax.set_xlabel("HLES (m)")
        ax.set_title(f"HLES distribution, {seas_name}")

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.set_title(common_params.varname_to_display_name[varname])
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5)
        # ax.text(1, 1, "(a)", fontdict=dict(weight="bold"), transform=ax.transAxes, va="top", ha="right")
        ax_with_legend = ax
        ax.set_xlim((0.1, None))

        # area average annual total HLES
        text_align_props = dict(transform=ax.transAxes, va="bottom", ha="right")


        # Plot the domain and the HLES region of interest
        # ax = fig.add_subplot(gs[0, 0])
        # topo_nc_file = data_root / "geophys_452x260_me.nc"
        # ax = plot_domain_and_interest_region(ax, topo_nc_file, focus_region_lonlat_nc_file=focus_region_lonlat_nc_file)
        # ax.set_title("(a) Experimental domain")
        #
        # # Add a common legend
        labels = list(label_to_handle)
        handles = [label_to_handle[l] for l in labels]
        ax_with_legend.legend(bbox_to_anchor=(0, -0.18), loc="upper left", borderaxespad=0., ncol=2)

        # ax.grid()
        sel_months_str = "_".join([str(m) for m in selected_months])

        common_params.img_folder.mkdir(exist_ok=True)
        img_file = common_params.img_folder / f"{varname}_histo_amount_cc_m{sel_months_str}_domain.png"
        print(f"Saving plot to {img_file}")
        fig.savefig(img_file, **common_params.image_file_options)


if __name__ == '__main__':
    # main(varname="hles_snow")

    for varname in ["hles_snow", ]:
        main(varname=varname)
