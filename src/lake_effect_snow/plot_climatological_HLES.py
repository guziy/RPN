from collections import OrderedDict
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

@main_decorator
def main():
    clevs_lkeff_snowfalldays = [0, 0.1, 0.8, 1.6, 2.4, 3.2, 4.0, 5]
    clevs_lkeff_snowfall = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 5, 10]
    mycolors = ["white", "indigo", "blue", "dodgerblue", "aqua", "lime", "yellow", "gold",
                                                     "orange", "orangered", "red", "firebrick", ]



    start_year = 1980
    end_year = 2009


    label_to_hles_dir = OrderedDict(
        [
         ("Obs", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_monthly_icefix_1980-2009")),
         ("CRCM5_NEMO", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2009")),
         ("CRCM5_HL", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_Hostetler_1980-2009")),
         # ("CRCM5_NEMO_TT_PR", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_based_on_TT_PR_1980-2009"))
        ]
    )


    label_to_line_style = {
        "Obs": "k.-",
        "CRCM5_NEMO": "r",
        "CRCM5_HL": "b",
        "CRCM5_NEMO_TT_PR": "g"
    }

    # clevs = clevs_lkeff_snowfall
    # vname = "snow_fall"
    # units = "cm"

    clevs = clevs_lkeff_snowfalldays
    vname = "lkeff_snowfall_days"
    units = "days"
    npc = 1


    cmap, bn = colors.from_levels_and_colors(clevs, mycolors[:len(clevs) - 1])
    cmap.set_over(mycolors[len(clevs) - 2])





    label_to_y_to_snfl = {}
    label_to_pc = {}

    label_to_eof = OrderedDict()
    label_to_varfraction = OrderedDict()

    mask = None

    plot_utils.apply_plot_params(font_size=12)


    years = None
    lats = None
    lons = None
    the_mask = None
    for label, folder in label_to_hles_dir.items():

        y_to_snfl = {}


        for the_file in folder.iterdir():
            if not the_file.name.endswith(".nc"):
                continue

            with Dataset(str(the_file)) as ds:
                print(ds)
                snfl = ds.variables[vname][:]
                year_current = ds.variables["year"][:]

                if mask is None:
                    lons, lats = [ds.variables[k][:] for k in ["lon", "lat"]]
                    lons[lons > 180] -= 360
                    mask = maskoceans(lons, lats, lons, inlands=True, resolution="i")


                if start_year <= year_current[0] <= end_year:
                    y_to_snfl[year_current[0]] = snfl[0]


        label_to_y_to_snfl[label] = y_to_snfl




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
    gs = GridSpec(1, len(label_to_hles_dir), wspace=0)


    for col, label in enumerate(label_to_hles_dir):

            y_to_snfl = label_to_y_to_snfl[label]

            snfl_clim = np.array([field for field in y_to_snfl.values()]).mean(axis=0)
            # snfl_clim = np.ma.masked_where(mask.mask, snfl_clim)

            ax = fig.add_subplot(gs[0, col])
            im = b.pcolormesh(xx, yy, snfl_clim, cmap=cmap, norm=bn, ax=ax)
            cb = b.colorbar(im, extend="max")
            cb.ax.set_visible(col == len(label_to_hles_dir) - 1)
            ax.set_title("{}".format(label))

            b.drawcoastlines(ax=ax)



    fig.savefig(str(label_to_hles_dir["Obs"].joinpath("hles_clim_{}_{}-{}.png".format(vname, start_year, end_year))), bbox_inches="tight", dpi=300)


if __name__ == '__main__':
    main()