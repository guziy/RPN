from pathlib import Path

import xarray

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.gridspec import GridSpec

from util import plot_utils

import pandas as pd

from matplotlib import dates


def plot_area_avg_CMOS_poster(data_dict: dict, bias_dict: dict, img_dir: Path, obs_label_hint="DAYMET",
                  panel_titles=(), plot_legend=True):

    img_dir.mkdir(parents=True, exist_ok=True)

    # calculate are avg
    ar_avgs = {}
    ar_avg_bias = {}

    times = None
    for data_key, data in data_dict.items():
        if times is None:
            times = data.coords["t"].values
            times = dates.date2num(pd.to_datetime(times).to_pydatetime())

        good_i, good_j = np.where(~data[0].to_masked_array().mask)
        ar_avgs[data_key] = data.values[:, good_i, good_j].mean(axis=1)
        if obs_label_hint not in data_key:
            ar_avg_bias[data_key] = bias_dict[data_key].values[:, good_i, good_j].mean(axis=1)

    plot_utils.apply_plot_params(font_size=8, width_cm=20, height_cm=5)
    fig = plt.figure()
    ax_list = []
    gs = GridSpec(1, 1)
    # ax = fig.add_subplot(gs[0, 0])
    # for data_key, data in ar_avgs.items():
    #     ax.plot(times, data, label=data_key.split("_ndrw")[0])
    #
    #
    # ax_list.append(ax)
    #
    # if len(panel_titles) > 0:
    #     ax.set_title(panel_titles[0])

    ax = fig.add_subplot(gs[0, 0])
    for data_key, data in ar_avgs.items():
        if obs_label_hint in data_key:
            continue
        ax.plot(times, ar_avg_bias[data_key], label="$\Delta$" + data_key.split("_ndrw")[0], lw=2)

    ax_list.append(ax)

    for pt, ax in zip(panel_titles, ax_list):
        ax.set_title(pt)

    for ax in ax_list:
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=1))
        if plot_legend:
            ax.legend()
        ax.grid(True, linestyle="--")

    imfile = img_dir / ("_".join([dl for dl in data_dict if obs_label_hint not in dl]) + "_CMOS_poster.png")
    fig.savefig(str(imfile), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    pass


if __name__ == '__main__':
    main()
