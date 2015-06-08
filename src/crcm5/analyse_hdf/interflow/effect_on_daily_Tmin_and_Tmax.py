from collections import OrderedDict
from matplotlib import cm
from matplotlib.gridspec import GridSpec

__author__ = 'huziy'

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

import matplotlib.pyplot as plt
import os
import numpy as np

img_folder = "images_for_lake-river_paper"


class CoordsAndBg:
    def __init__(self, lons2d, lats2d, bmp):
        self.basemap = bmp
        self.lons2d = lons2d
        self.lats2d = lats2d

        self.x = None
        self.y = None

    def get_xy(self):
        if self.x is None or self.y is None:
            self.x, self.y = self.basemap(self.lons2d, self.lats2d)
        return self.x, self.y


def plot_impacts_of_intfl_on_seasonal_means(var_name="Tmin", base_label="",
                                            base_data_daily=None, label_to_data_daily=None,
                                            season_to_months=None, daily_dates=None,
                                            coord_obj=None):
    assert isinstance(label_to_data_daily, dict)
    assert isinstance(season_to_months, dict)
    assert isinstance(coord_obj, CoordsAndBg)

    labels_joined = "-".join(sorted(label_to_data_daily.keys()))
    current_img_folder = "{}_vs_{}".format(labels_joined, base_label)
    current_img_folder = os.path.join(img_folder, current_img_folder)

    if not os.path.isdir(current_img_folder):
        os.makedirs(current_img_folder)

    seasons = "-".join(sorted(season_to_months.keys()))

    img_file = "{}_{}.png".format(var_name, seasons)
    img_file = os.path.join(current_img_folder, img_file)

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(len(label_to_data_daily) + 1, 5, width_ratios=[1, 1, 1, 1, 0.05])

    x, y = coord_obj.get_xy()
    bmp = coord_obj.basemap


    # for values
    vmin = -30
    vmax = 30
    cmap = cm.get_cmap("jet", 10)

    # Plot the base values
    row = 0
    col = 0
    im = None
    for season, months in season_to_months.items():
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season)

        the_mean = np.mean(list(f for t, f in zip(daily_dates, base_data_daily) if t.month in months),
                           axis=0)

        im = bmp.pcolormesh(x, y, the_mean, cmap=cmap, vmin=vmin, vmax=vmax)
        bmp.drawcoastlines()
        if not col:
            ax.set_ylabel(base_label)

        col += 1
    plt.colorbar(im, cax=fig.add_subplot(gs[row, -1]))
    row += 1

    # Plot differences
    cmap = cm.get_cmap("RdBu_r", 10)
    vmin = -0.5
    vmax = 0.5
    for label, data_daily in label_to_data_daily.items():
        col = 0
        for season, months in season_to_months.items():
            ax = fig.add_subplot(gs[row, col])
            if not col:
                ax.set_ylabel(r"$\Delta$" + "{}".format(label))

            the_mean = np.mean(list(f - b for t, f, b in zip(daily_dates, data_daily, base_data_daily)
                                    if t.month in months), axis=0)
            im = bmp.pcolormesh(x, y, the_mean, cmap=cmap, vmin=vmin, vmax=vmax)
            bmp.drawcoastlines()
            col += 1

        row += 1

    plt.colorbar(im, cax=fig.add_subplot(gs[1:, -1]))

    fig.savefig(img_file)
    plt.close(fig)


def main():
    start_year = 1980
    end_year = 2010

    base_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    base_label = "CRCM5-L"

    label_to_path = {
        "CRCM5-LI": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5"
    }

    season_to_months = OrderedDict([
        ("Winter", [12, 1, 2]),
        ("Spring", list(range(3, 6))),
        ("Summer", list(range(6, 9))),
        ("Fall", list(range(9, 12)))
    ])

    coords = CoordsAndBg(*analysis.get_basemap_from_hdf(base_path))

    daily_dates, tmax_daily_base = analysis.get_daily_max_climatology(base_path, var_name="TT_max", level=0,
                                                                      start_year=start_year, end_year=end_year)

    _, tmin_daily_base = analysis.get_daily_min_climatology(base_path, var_name="TT_min", level=0,
                                                            start_year=start_year, end_year=end_year)

    label_to_tmax_daily = {}
    label_to_tmin_daily = {}

    for label, the_path in label_to_path.items():
        _, label_to_tmax_daily[label] = analysis.get_daily_max_climatology(the_path, var_name="TT_max", level=0,
                                                                           start_year=start_year, end_year=end_year)

        _, label_to_tmin_daily[label] = analysis.get_daily_min_climatology(the_path, var_name="TT_min", level=0,
                                                                           start_year=start_year, end_year=end_year)


    # Plot results
    plot_impacts_of_intfl_on_seasonal_means(var_name="Tmin", base_label=base_label,
                                            base_data_daily=tmin_daily_base,
                                            label_to_data_daily=label_to_tmin_daily,
                                            coord_obj=coords, season_to_months=season_to_months,
                                            daily_dates=daily_dates)

    plot_impacts_of_intfl_on_seasonal_means(var_name="Tmax", base_label=base_label,
                                            base_data_daily=tmax_daily_base,
                                            label_to_data_daily=label_to_tmax_daily,
                                            coord_obj=coords, season_to_months=season_to_months,
                                            daily_dates=daily_dates)


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    main()
