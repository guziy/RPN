import os
from pathlib import Path
from matplotlib import cm
from matplotlib.dates import date2num, num2date, DateFormatter, DateLocator, MonthLocator
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from nemo.obs import AdcpProfileObs
from util import plot_utils

__author__ = 'san'

import numpy as np

import matplotlib.pyplot as plt

from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager


def get_img_folder():
    img_dir = Path("nemo/adcp_comparisons")
    if not img_dir.is_dir():
        img_dir.mkdir(parents=True)


FLOW_SPEED = "flow speed"
FLOW_DIRECTION = "flow direction"

def plot_profiles():
    obs_base_dir = Path("/home/huziy/skynet3_rech1/nemo_obs_for_validation/data_from_Ram_Yerubandi/ADCP-profiles")
    obs_dir_list = [
        str(obs_base_dir.joinpath("105.317")),
        str(obs_base_dir.joinpath("155.289"))
    ]

    obs_var_col = AdcpProfileObs.vmag_col
    model_var_name = FLOW_SPEED

    model_folder = "/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012"

    manager_nemo_u = NemoYearlyFilesManager(folder=model_folder, suffix="_U.nc")
    manager_nemo_v = NemoYearlyFilesManager(folder=model_folder, suffix="_V.nc")
    manager_nemo_w = NemoYearlyFilesManager(folder=model_folder, suffix="_W.nc")



    fig = plt.figure()
    gs = GridSpec(len(obs_dir_list), 5, width_ratios=[1, 1, 0.05, 1, 0.05])


    cmap = cm.get_cmap("jet", 10)
    diff_cmap = cm.get_cmap("RdBu_r", 10)

    for i, obs_dir in enumerate(obs_dir_list):

        adcp = AdcpProfileObs()

        dates, levels, obs_data = adcp.get_acdp_profiles(folder=obs_dir, data_column=obs_var_col)

        dates_m, levs_m, u_cs = manager_nemo_u.get_tz_crosssection_for_the_point(
            lon=adcp.longitude, lat=adcp.latitude,
            start_date=dates[0], end_date=dates[-1],
            var_name="vozocrtx", zlist=levels
        )

        dates_m, levs_m, v_cs = manager_nemo_v.get_tz_crosssection_for_the_point(
            lon=adcp.longitude, lat=adcp.latitude,
            start_date=dates[0], end_date=dates[-1],
            var_name="vomecrty", zlist=levels
        )

        dates_m, levs_m, w_cs = manager_nemo_w.get_tz_crosssection_for_the_point(
            lon=adcp.longitude, lat=adcp.latitude,
            start_date=dates[0], end_date=dates[-1],
            var_name="vovecrtz", zlist=levels
        )




        numdates = date2num(dates.tolist())
        print("Obs dates are: {} ... {}".format(dates[0], dates[-1]))
        print([num2date([n for n in numdates if n not in dates_m])])
        zz, tt = np.meshgrid(levels, numdates)

        umag_mod_cs = (u_cs ** 2 + v_cs ** 2 + w_cs ** 2) ** 0.5 * 100.0

        all_axes = []
        # Obs
        ax = fig.add_subplot(gs[i, 0])
        ax.set_title("Obs")
        cs = ax.contourf(tt, zz, obs_data, cmap=cmap)
        ax.set_ylabel("({:.1f}, {:.1f})".format(adcp.longitude, adcp.latitude))
        all_axes.append(ax)


        # Model
        ax = fig.add_subplot(gs[i, 1])
        ax.set_title("NEMO-offline")
        cs = ax.contourf(tt, zz, umag_mod_cs, levels=cs.levels, cmap=cmap)
        all_axes.append(ax)

        plt.colorbar(cs, cax=fig.add_subplot(gs[i, 2]))

        # Bias
        ax = fig.add_subplot(gs[i, 3])
        ax.set_title("Model - Obs.")
        delta = umag_mod_cs - obs_data

        vmax = np.abs(delta).max()
        vmin = -vmax
        locator = MaxNLocator(nbins=diff_cmap.N, symmetric=True)

        cs = ax.contourf(tt, zz, delta, levels=locator.tick_values(vmin, vmax), cmap=diff_cmap)
        plt.colorbar(cs, cax=fig.add_subplot(gs[i, 4]))
        all_axes.append(ax)


        for the_ax in all_axes:
            the_ax.xaxis.set_major_formatter(DateFormatter("%b"))
            the_ax.xaxis.set_major_locator(MonthLocator())
            the_ax.invert_yaxis()

    img_folder = Path("nemo/adcp")
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    img_file = img_folder.joinpath("adcp_profiles.pdf")
    fig.tight_layout()
    fig.savefig(str(img_file), bbox_inches="tight")



def plot_point_positions():
    fig = plt.figure()



def main():
    plot_profiles()



if __name__ == '__main__':
    import application_properties
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(font_size=12, width_cm=30)
    main()