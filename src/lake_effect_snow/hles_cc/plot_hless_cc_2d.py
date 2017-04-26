
import matplotlib
matplotlib.use("Agg")


from collections import OrderedDict
from collections import defaultdict
from pathlib import Path

from eofs.standard import Eof
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans, Basemap

from application_properties import main_decorator
from lake_effect_snow import common_params
from util import plot_utils
import matplotlib.pyplot as plt
from netCDF4 import Dataset

from matplotlib import colors

import numpy as np

import pandas as pd

import re


def _get_year_and_month_from_filename(fname:str):
    """
    fname example: CRCM5_NEMO_CanESM2_RCP85_2079-2084_lkeff_snfl_2084-2084_m12-12.nc
    :param fname:
    """


    fields = fname.split("_")

    m1, m2 = [int(m) for m in re.findall("\d+", fields[-1])]
    y1, y2 = [int(m) for m in re.findall("\d+", fields[-2])]

    assert m1 == m2
    assert y1 == y2

    return y1, m1


def get_year_month_to_filepath_map(data_dir: Path):

    ym_to_path = {}

    for f in data_dir.iterdir():
        if not f.name.lower().endswith(".nc"):
            continue


        y, m = _get_year_and_month_from_filename(f.name)

        ym_to_path[y, m] = str(f)

    return ym_to_path





def read_var_from_hles_alg_output(folder_path: Path, varname: str, start_year: int, end_year: int,
                                  start_month:int, nmonths:int=1) -> tuple:


    """
    return the  (mean, std, nfields)
    :param folder_path:
    :param varname:
    :param start_year:
    :param end_year:
    :param start_month:
    :param nmonths:
    :return:
    """

    ym_to_path = get_year_month_to_filepath_map(folder_path)

    y_to_fields = defaultdict(list)

    for y in range(start_year, end_year + 1):
        for month in range(start_month, start_month + nmonths):

            cy = y + int(month > 12)
            cm = month % 13 + int(month > 12)

            with Dataset(ym_to_path[cy, cm]) as ds:
                print(ym_to_path[cy, cm], ds.variables[varname][:].shape)
                y_to_fields[y].append(ds.variables[varname][:].squeeze())

    fields3d = np.ma.array([field for field in y_to_fields.values()])
    print(fields3d.shape)


    the_mean = fields3d.mean(axis=0)
    the_std = fields3d.std(axis=0)
    nobs = fields3d.shape[0]

    return the_mean, the_std, nobs


def get_lons_and_lats(data_folder:Path):

    lons, lats = None, None
    for f in data_folder.iterdir():
        if f.name.lower().endswith(".nc"):
            with Dataset(str(f)) as ds:
                lons, lats = [ds.variables[k][:] for k in ["lon", "lat"]]

    return lons, lats


@main_decorator
def main():

    image_dir = Path("climate_change_hles")
    if not image_dir.exists():
        image_dir.mkdir()

    clevs_lkeff_snowfalldays = [0, 0.1, 0.8, 1.6, 2.4, 3.2, 4.0, 5]
    clevs_lkeff_snowfall = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10]
    mycolors = ["white", "indigo", "blue", "dodgerblue", "aqua", "lime", "yellow", "gold",
                                                     "orange", "orangered", "red", "firebrick", ]



    clevs = clevs_lkeff_snowfalldays
    vname = "lkeff_snowfall_days"
    units = "days"

    hles_start_month = 11
    hles_nmonths = 3


    label_current = "CRCM5_NEMO_Current"
    start_year_current = 1989
    end_year_current = 1994
    period_current = (start_year_current, end_year_current)



    label_future = "CRCM5_NEMO_Future"
    start_year_future = 2079
    end_year_future = 2083
    period_future = (start_year_future, end_year_future)


    label_to_hles_dir = OrderedDict(
        [
         (label_current, Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_1989-1995_1989-1995")),
         (label_future, Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_CanESM2_RCP85_2079-2084_2079-2084")),
        ]
    )

    label_to_period = {
        label_current: period_current,
        label_future: period_future
    }




    # read data and calculate means
    label_to_data = {}
    for col, label in enumerate(label_to_hles_dir):

        start_year, end_year = label_to_period[label]

        label_to_data[label] = read_var_from_hles_alg_output(label_to_hles_dir[label], varname=vname,
                                                             start_year=start_year, end_year=end_year,
                                                             start_month=hles_start_month, nmonths=hles_nmonths)





    # plotting ...


    cmap, bn = colors.from_levels_and_colors(clevs, mycolors[:len(clevs) - 1])
    cmap.set_over(mycolors[len(clevs) - 2])




    plot_utils.apply_plot_params(font_size=12)

    lons, lats = get_lons_and_lats(label_to_hles_dir[label_current])
    lons[lons < 0] += 360
    b = Basemap(lon_0=180,
                llcrnrlon=lons[0, 0],
                llcrnrlat=lats[0, 0],
                urcrnrlon=lons[-1, -1],
                urcrnrlat=lats[-1, -1],
                resolution="i", area_thresh=2000)



    # plot the eofs


    plot_utils.apply_plot_params(font_size=10, width_cm=30, height_cm=8)

    xx, yy = b(lons, lats)

    fig = plt.figure()
    gs = GridSpec(1, len(label_to_hles_dir) + 1, wspace=0)

    for col, label in enumerate(label_to_hles_dir):
            the_mean, the_std, nobs = label_to_data[label]

            # snfl_clim = np.ma.masked_where(mask.mask, snfl_clim)

            ax = fig.add_subplot(gs[0, col])
            im = b.pcolormesh(xx, yy, the_mean, cmap=cmap, norm=bn, ax=ax)
            cb = b.colorbar(im, extend="max")
            cb.ax.set_visible(False)
            ax.set_title("{}".format(label))
            b.drawcoastlines(ax=ax)


    # plot the cc
    ax = fig.add_subplot(gs[0, -1])
    the_mean = label_to_data[label_future][0] - label_to_data[label_current][0]
    im = b.pcolormesh(xx, yy, the_mean, cmap=cmap, norm=bn, ax=ax)
    cb = b.colorbar(im, extend="max")
    cb.ax.set_visible(False)
    ax.set_title("F - C")
    b.drawcoastlines(ax=ax)

    period_current_s = "{}-{}".format(*period_current)
    period_future_s = "{}-{}".format(*period_future)

    fig.savefig(str(image_dir.joinpath("hles_cs_{}_{}_{}.png".format(vname, period_current_s, period_future_s))), bbox_inches="tight", dpi=300)



def test():
    _get_year_and_month_from_filename("CRCM5_NEMO_CanESM2_RCP85_2079-2084_lkeff_snfl_2084-2084_m12-12.nc")

if __name__ == '__main__':
    main()
