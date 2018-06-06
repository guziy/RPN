
from pathlib import Path

import xarray

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap, maskoceans

from util import plot_utils

import pandas as pd

from matplotlib import dates, cm, colors


def create_ml_polygon(reg_lons, reg_lats, bmap):
    x_ll, y_ll = bmap(reg_lons[0, 0], reg_lats[0, 0])
    x_ur, y_ur = bmap(reg_lons[-1, -1], reg_lats[-1, -1])

    return Polygon([(x_ll, y_ll), (x_ll, y_ur), (x_ur, y_ur), (x_ur, y_ll)], fc="none", edgecolor="k", linewidth=2)



def plot_meridional_mean(data_dict: dict,
                         bias_dict:dict,
                         img_dir: Path,
                         obs_label_hint="DAYMET",
                         panel_titles=(),
                         bmap: Basemap=None,
                         months=None,
                         season_name="annual",
                         meridional_elev_dict=None, map_topo=None,
                         plot_values=True, plot_legend=True, lon_min=None, lon_max=None
                         ):

    """
    Expect the data to be already selected in space, and the average along the latitude axis (assumed to be the last) is calculated
    before plotting
    :param map_topo: topo map
    :param season_name: season name goes together
    :param months: if None the annual mean is calculated, otherwize the mean is calculated over the list of months
    :param bmap:
    :param data_dict:
    :param bias_dict:
    :param img_dir:
    :param obs_label_hint:
    :param panel_titles:
    """
    legend_fontsize = 14
    line_width=3

    img_dir.mkdir(parents=True, exist_ok=True)


    # calculate are avg
    meridional_avgs = {}
    meridional_avg_bias = {}
    meridional_avg_lons = {}

    times = None
    time_sel_vec = None
    data_with_common_meta = None

    for data_key, data in data_dict.items():
        if times is None:
            times = data.coords["t"].values
            times = dates.date2num(pd.to_datetime(times).to_pydatetime())
            data_with_common_meta = data

        # calculate the meridional average of the lon filed
        meridional_avg_lons[data_key] = data.coords["lon"].values.mean(axis=1)

        meridional_avgs[data_key] = data.values.mean(axis=2)

        if obs_label_hint not in data_key:
            meridional_avg_bias[data_key] = bias_dict[data_key].values.mean(axis=2)


        # month selection
        if time_sel_vec is None:
            if months is None:
                time_sel_vec = np.ones((len(times),), dtype=bool)
            else:
                all_months = data.coords["t.month"].values
                time_sel_vec = np.asarray([m in months for m in all_months])



    # plotting
    plot_utils.apply_plot_params(font_size=14, height_cm=3)

    fig = plt.figure()

    ax_list = []
    height_ratios = [2, 2, 1] if plot_values else [2, 1]
    nrows = len(height_ratios)

    gs = GridSpec(nrows, 2, width_ratios=[2, 1], height_ratios=height_ratios, wspace=0.01, hspace=0.05)
    # plot values
    if plot_values:
        ax = fig.add_subplot(gs[0, 0])
        for data_key, data in meridional_avgs.items():
            ax.plot(meridional_avg_lons[data_key], data[time_sel_vec, :].mean(axis=0), label=data_key.split("_ndrw")[0],
                    lw=line_width)

        ax.yaxis.set_label_position("right")
        ax.set_ylabel(f"{data_with_common_meta.units}", rotation=270, va="bottom")
        ax_list.append(ax)
    else:
        ax = None


    # plot biases
    row = int(plot_values)
    ax = fig.add_subplot(gs[row, 0], sharex=ax)
    for data_key, data in meridional_avgs.items():
        if obs_label_hint in data_key:
            continue

        ax.plot(meridional_avg_lons[data_key], meridional_avg_bias[data_key][time_sel_vec, :].mean(axis=0),
                label="$\Delta$" + data_key.split("_ndrw")[0], lw=line_width)

    ax.yaxis.set_label_position("right")
    ax.set_ylabel(f"bias, {data_with_common_meta.units}", rotation=270, va="bottom")
    ax_list.append(ax)

    # plot the elevation
    if meridional_elev_dict is not None:
        row += 1
        ax = fig.add_subplot(gs[row, 0], sharex=ax)
        for data_key, elev in meridional_elev_dict.items():
            ax.plot(elev.coords["lon"], elev.values, label=data_key, lw=line_width)

        if plot_legend:
            ax.legend(fontsize=legend_fontsize)

        ax.set_xlabel("Longitude")
        ax.grid(True, linestyle="--")
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Elevation, m", rotation=270, va="bottom")
        if lon_min is not None and (lon_max is not None):
            ax.set_xlim(lon_min, lon_max)




    # plot the map

    if map_topo is not None:
        ax = fig.add_subplot(gs[0, 1])
        bmap.drawmapboundary(fill_color="0.4", ax=ax)
        bmap.drawcoastlines(ax=ax, linewidth=0.1)
        bmap.drawcountries(linewidth=0.1)
        bmap.drawstates(linewidth=0.1)


        data_random = list(data_dict.items())[0][1]
        ax.add_patch(
            create_ml_polygon(data_random.coords["lon"], data_random.coords["lat"], bmap)
        )

        topo_levels = [0, 500, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000]
        bn = BoundaryNorm(topo_levels, len(topo_levels) - 1)
        cmap = cm.get_cmap("terrain", len(topo_levels) - 1)
        cmap = colors.LinearSegmentedColormap.from_list("topo_cut", cmap(np.arange(0.4, 1.1, 0.1)), N=len(topo_levels) - 1)

        lons, lats = map_topo.coords["lons"].values, map_topo.coords["lats"].values
        xx, yy = bmap(lons, lats)

        to_plot = maskoceans(np.where(lons > 180, lons - 360, lons), lats, map_topo.values, inlands=True)
        im = bmap.pcolormesh(xx, yy, to_plot, cmap=cmap, norm=bn)
        bmap.colorbar(im)

    for pt, ax in zip(panel_titles, ax_list):
        ax.set_title(pt)

    for ax in ax_list:
        ax.set_xlabel("Longitude")
        ax.grid(True, linestyle="--")

        # Hide ticks in the 2 upper panels
        for the_label in ax.get_xticklabels():
            the_label.set_visible(False)

        # Hide the xaxis label of the 2 upper panels
        ax.xaxis.get_label().set_visible(False)
        if lon_min is not None and (lon_max is not None):
            ax.set_xlim(lon_min, lon_max)



    imfile = img_dir / ("_".join([dl for dl in data_dict if obs_label_hint not in dl]) + f"_{season_name}.png")
    fig.savefig(str(imfile), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    pass


if __name__ == '__main__':
    main()
