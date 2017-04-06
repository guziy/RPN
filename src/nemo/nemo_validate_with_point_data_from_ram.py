from pathlib import Path

from matplotlib import cm
from matplotlib.dates import DateFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from util import plot_utils

__author__ = 'huziy'

import os
from nemo import obs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Validate temperature profiles, flow profiles at given points

# ADCP data are on different levels: level values are encoded in the file names
# Zhao Jun: "the last four number of '021A0XXXX' is the depth in cm."
#


def plot_flow_speed_profiles_comparisons():

    file_prefixes = [

    ]
    pass


img_dir = Path("nemo/nemo_coupled")

def main_plot_all_temp_profiles_in_one_figure(
        folder_path="/home/huziy/skynet3_rech1/nemo_obs_for_validation/data_from_Ram_Yerubandi/T-profiles"):
    """

    figure layout:
        Obs     Model  (Model - Obs)
    p1

    p2

    ...

    pn

            Point Positions map

    :param folder_path: Path to the text files containing observed temperature profiles
    """
    folder_path = os.path.expanduser(folder_path)

    temperature_profile_file_prefixes = [
        "08-01T-004A024.120.290",
        # "08-01T-013A054.120.290", "08-00T-012A017.105.317",
        "08-00T-004A177.106.286"
    ]

    # nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012")
    # nemo_manager = NemoYearlyFilesManager(folder="/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/zdf_gls_dt_and_sbc_5min")
    nemo_manager = NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO")

    plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=35, height_cm=30)
    fig = plt.figure()
    gs = GridSpec(len(temperature_profile_file_prefixes) + 2, 4, width_ratios=[1, 1, 1, 0.05],
                  height_ratios=len(temperature_profile_file_prefixes) * [1.0, ] + [0.05, 2, ], top=0.90,
                  wspace=0.2, hspace=0.2)

    color_levels = np.arange(0, 30, 0.5)
    diff_levels = np.arange(-7, 8, 2)
    diff_cmap = cm.get_cmap("bwr", len(diff_levels) - 1)
    axes_list = []
    imvalues = None
    imdiff = None
    titles = ["Obs", "Model", "Model - Obs"]
    labels = ["P{}".format(p) for p in range(len(temperature_profile_file_prefixes))]


    nupper_levs_to_plot = 5

    obs_point_list = []
    start_date, end_date = None, None
    for row, prefix in enumerate(temperature_profile_file_prefixes):
        # Get the data for plots
        po = obs.get_profile_for_prefix(prefix, folder=folder_path)
        obs_point_list.append(po)

        tto, zzo, obs_profile = po.get_tz_section_data()

        start_date = po.get_start_date()
        end_date = po.get_end_date()

        tt, zz, model_profile_interp = nemo_manager.get_tz_crosssection_for_the_point(lon=po.longitude, lat=po.latitude,
                                                                                      zlist=po.levels,
                                                                                      var_name="votemper",
                                                                                      start_date=start_date,
                                                                                      end_date=end_date)

        ttm, zzm, model_profile = nemo_manager.get_tz_crosssection_for_the_point(lon=po.longitude, lat=po.latitude,
                                                                                 zlist=None,
                                                                                 var_name="votemper",
                                                                                 start_date=start_date,
                                                                                 end_date=end_date)

        print("Unique model levels: ", np.unique(zzm))

        # obs
        ax = fig.add_subplot(gs[row, 0])
        ax.contourf(tto, zzo, obs_profile, levels=color_levels)
        axes_list.append(ax)
        ax.set_ylabel(labels[row])
        common_ylims = ax.get_ylim()

        # model (not interpolated)
        ax = fig.add_subplot(gs[row, 1])
        imvalues = ax.contourf(ttm, zzm, model_profile, levels=color_levels)
        ax.set_ylim(common_ylims)
        ax.yaxis.set_ticklabels([])
        axes_list.append(ax)



        # model profile (interpolated to the observation levels) - obs profile
        ax = fig.add_subplot(gs[row, 2])
        diff = model_profile_interp - obs_profile
        imdiff = ax.contourf(tt[:, :nupper_levs_to_plot], zz[:, :nupper_levs_to_plot], diff[:, :nupper_levs_to_plot], levels=diff_levels, cmap=diff_cmap, extend="both")
        # ax.yaxis.set_ticklabels([])
        axes_list.append(ax)

        if not row:
            for axi, title in zip(axes_list, titles):
                axi.set_title(title)


    title_suffix = " {}-{}".format(start_date.year, end_date.year) if start_date.year < end_date.year \
        else " {}".format(start_date.year)
    fig.suptitle("Temperature, " + title_suffix, font_properties=FontProperties(weight="bold"))


    # plot colorbars
    cb = plt.colorbar(imvalues, cax=fig.add_subplot(gs[len(temperature_profile_file_prefixes), :2]), orientation="horizontal")

    plt.colorbar(imdiff, cax=fig.add_subplot(gs[:len(temperature_profile_file_prefixes), 3]))

    # Format dates
    dfmt = DateFormatter("%b")
    for i, ax in enumerate(axes_list):
        ax.xaxis.set_major_formatter(dfmt)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.invert_yaxis()


    for ax in axes_list[:-3]:
        ax.xaxis.set_ticklabels([])


    # Plot station positions
    lons, lats, bmp = nemo_manager.get_coords_and_basemap(resolution="i")
    ax = fig.add_subplot(gs[len(temperature_profile_file_prefixes) + 1, :-2])

    for i, po, label in zip(list(range(len(obs_point_list))), obs_point_list, labels):
        xx, yy = bmp(po.longitude, po.latitude)
        bmp.scatter(xx, yy, c="r")

        multiplier = 0.5 if i % 2 else 1
        if i > 1:
            multiplier *= -4

        ax.annotate(label, xy=(xx, yy), xytext=(-20 * multiplier, 20 * multiplier),
                    textcoords='offset points', ha='right', va='bottom',
                    font_properties=FontProperties(size=14),
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    bmp.drawcoastlines(linewidth=0.5, ax=ax)
    # fig.tight_layout()


    if not img_dir.exists():
        img_dir.mkdir(parents=True)

    img_path = img_dir / "T-profiles.pdf"

    print("Saving plots to {}".format(img_path))
    fig.savefig(str(img_path), transparent=True, bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    main_plot_all_temp_profiles_in_one_figure()