import calendar
from pathlib import Path

import xarray
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap

from util import plot_utils
import matplotlib.pyplot as plt


def __get_ij_from_index(ind, ncols, nrows):
    i = ind // nrows
    j = ind % nrows
    return j, i

def plot_monthly_panels(data:xarray.DataArray, basemap: Basemap, img_dir="default_img_dir",
                        data_label="",
                        color_levels=None, cmap=cm.get_cmap("jet")):
    """

    :param data:
    :param basemap:
    :param img_dir:
    :param data_label: should contain period and simulation and obs sources used to get the data
    :param color_levels:
    :param cmap:
    """

    plot_utils.apply_plot_params(font_size=14, width_cm=30)

    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)


    xx, yy = basemap(data["lon"].values, data["lat"].values)


    # calculate monthly means:
    monthly_data = data.groupby("t.month").mean(dim="t")

    nrows = 3
    ncols = 4

    gs = GridSpec(nrows=nrows, ncols=ncols, wspace=0.0)
    fig = plt.figure()


    norm = BoundaryNorm(color_levels, len(color_levels) - 1)
    cmap = cm.get_cmap(cmap, len(color_levels) - 1)

    month_with_colorbar = 2
    month_with_data_label = 5
    month_with_big_data_label = 2

    for ind, month in enumerate([12, ] + list(range(1, 12))):
        i, j = __get_ij_from_index(ind, ncols, nrows)
        ax = fig.add_subplot(gs[i, j])

        ax.set_title(calendar.month_abbr[month])

        im = basemap.pcolormesh(xx, yy,
                monthly_data.sel(month=month).to_masked_array(), cmap=cmap, norm=norm, ax=ax)
        basemap.drawcoastlines(ax=ax)
        basemap.drawstates(ax=ax, linewidth=0.5)
        basemap.drawcountries(ax=ax, linewidth=0.5)

        cb = basemap.colorbar(im, location="bottom")

        cb.ax.set_visible(month == month_with_colorbar) # show only the colorbar for October

        if month == month_with_colorbar:
            cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=45)


        if month == month_with_data_label:
            ax.set_xlabel("ndrw" + data_label.split("_ndrw")[1], ha="left", fontsize=10)

        if month == month_with_big_data_label:
            ax.set_ylabel(data_label.split("_ndrw")[0], ha="left")

    # save the plot to a file
    img_path = img_dir / f"{data_label}.png"
    fig.savefig(str(img_path), dpi=400, bbox_inches="tight")
    plt.close(fig)




def main():
    # TODO: write tests
    pass


if __name__ == '__main__':
    main()
