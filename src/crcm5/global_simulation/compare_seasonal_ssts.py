import os
from collections import OrderedDict
from pathlib import Path

from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap, maskoceans
from application_properties import main_decorator
from crcm5.global_simulation import commons

from rpn.rpn_multi import MultiRPN
import re

import numpy as np
import matplotlib.pyplot as plt

from util import plot_utils
from mpl_toolkits.basemap import cm as basemap_cm


def plot_row(row=0, season_to_mean=None, difference=False, label="", bmp=None, xx=None, yy=None,
             axes=None, clevs=None):

    cs = None
    cmap = cm.get_cmap("bwr") if difference else basemap_cm.GMT_wysiwyg

    for col, (season, field) in enumerate(season_to_mean.items()):

        ax = axes[col]



        # Make sure that colormaps are the same in all columns of the row
        if cs is not None:
            clevs = cs.levels
        elif clevs is None:
            clevs = 20
        plt.sca(ax)
        cs = bmp.contourf(xx, yy, field, clevs, ax=ax,
                          extend="both", cmap=cmap)


        bmp.drawcoastlines(linewidth=0.3, ax=ax)

        bmp.drawmapboundary(fill_color="0.75", ax=ax)

        if col == 0:
            ax.set_ylabel(label)

        if row == 0:
            ax.set_title(season)

    cb = plt.colorbar(cs, cax=axes[-1])
    return cs.levels



def _get_month_and_year_from_file_name(fname):
    for match in re.finditer("(\d{6,})", fname):
        ym = match.groups()[0]
        return int(ym[-2:]), int(ym[:-2])


@main_decorator
def main():
    plot_utils.apply_plot_params(width_cm=40, height_cm=25, font_size=14)
    seasons = commons.default_seasons
    dx_plotting = 0.5  # degrees

    # units = "$^\circ$C"
    # long_name = "SST, {}".format(units)
    # var_name = "TM"
    # clevs_diff = np.arange(-5, 5.5, 0.5)


    units = ""
    long_name = "Sea ice {}".format(units)
    var_name = "LG"
    clevs_diff = [v for v in np.arange(-1, 1.1, 0.1) if abs(v) > 1.0e-6]


    label_to_folder = OrderedDict([
        ("ERA-Interim", "/home/huziy/skynet3_rech1/CNRCWP/Calgary_flood/SST_SeaIce/I_SST_SeaIce"),
        ("PreI-CanESM2", "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/SST_SeaIce/PreI_SST_SeaIce"),
        ("PreI-GFDL", "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/SST_SeaIce/PreI_SST_SeaIce"),
        ("PreI-GISS", "/RESCUE/skynet3_rech1/huziy/CNRCWP/Calgary_flood/SST_SeaIce/PreI_SST_SeaIce")
    ])




    bmp = Basemap(projection="robin", lon_0=180)
    xx, yy = None, None

    year = 2013

    print(seasons)

    label_to_season_to_field = OrderedDict()
    for label, folder in label_to_folder.items():
        label_to_season_to_field[label] = OrderedDict()
        all_files = [f for f in os.listdir(folder) if label.lower() in f.lower() or label.split("-")[1].lower() in f.lower()]

        for sname, months in seasons.items():
            season_files = [f for f in all_files if _get_month_and_year_from_file_name(f)[0] in months and _get_month_and_year_from_file_name(f)[1] == year]

            season_files = [os.path.join(folder, f) for f in season_files]

            print(10 * "+" + sname + "+" * 10)
            print(season_files)

            r = MultiRPN(season_files)

            data = r.get_4d_field(var_name)

            the_mean = np.array([list(v.items())[0][1] for d, v in data.items()]).mean(axis=0)


            lons, lats = r.get_longitudes_and_latitudes_of_the_last_read_rec()
            print("lon range", lons.min(), lons.max())
            print("lat range", lats.min(), lats.max())

            # lons, lats, the_mean = commons.interpolate_to_uniform_global_grid(the_mean, lons_in=lons, lats_in=lats, out_dx=dx_plotting)

            # mask land
            lons1 = lons.copy()
            lons1[lons1 > 180] -= 360
            ocean_mask = maskoceans(lons1, lats, np.zeros_like(the_mean))
            the_mean = np.ma.masked_where(~ocean_mask.mask, the_mean)

            if xx is None:
                xx, yy = bmp(lons, lats)


            label_to_season_to_field[label][sname] = the_mean



    # plotting------

    fig = plt.figure()

    gs = GridSpec(len(label_to_season_to_field), len(seasons) + 1, width_ratios=[1.0, ] * len(seasons) + [0.05, ])

    #
    base_fields = None
    base_label = None

    for row, (label, season_to_mean) in enumerate(label_to_season_to_field.items()):

        print(gs.get_geometry())

        axes = [fig.add_subplot(gs[row, col]) for col in range(len(season_to_mean) + 1)]

        common_params = dict(axes=axes, bmp=bmp, xx=xx, yy=yy)
        if base_fields is None:
            plot_row(row, season_to_mean=season_to_mean, label=label, **common_params)
            base_fields = season_to_mean
            base_label = label
        else:
            plot_row(row, season_to_mean=OrderedDict([(k, v - base_fields[k]) for k, v in season_to_mean.items()]),
                     difference=True, clevs=clevs_diff, label="{}\n-\n{}".format(label, base_label),
                     **common_params)


    fig.suptitle(long_name)
    img_folder = Path("industrial_and_preindustrial_sst").joinpath("seasonal")
    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    img_format = "png"
    img_file = img_folder.joinpath("{}.{}".format(var_name, img_format))

    with img_file.open("wb") as f:
        fig.savefig(f, bbox_inches="tight", format=img_format)






if __name__ == '__main__':
    main()
