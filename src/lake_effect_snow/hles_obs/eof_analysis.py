from collections import OrderedDict
from pathlib import Path

from eofs.standard import Eof
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans

from application_properties import main_decorator
from util import plot_utils
import matplotlib.pyplot as plt
from netCDF4 import Dataset

import numpy as np

import pandas as pd

@main_decorator
def main():
    folder_path = "/HOME/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_1980-2009"



    label_to_hles_dir = OrderedDict(
        [
         ("Obs", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_Obs_1980-2009")),
         ("CRCM5_NEMO", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_NEMO_1980-2009")),
         ("CRCM5_HL", Path("/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Python/RPN/lake_effect_analysis_CRCM5_Hostetler_1980-2009"))
        ]
    )


    label_to_line_style = {
        "Obs": "k.-",
        "CRCM5_NEMO": "r",
        "CRCM5_HL": "b"
    }

    vname = "snow_fall"
    # vname = "lkeff_snowfall_days"
    units = "cm"
    npc = 1

    label_to_y_to_snfl = {}
    label_to_pc = {}

    label_to_eof = OrderedDict()
    label_to_varfraction = OrderedDict()

    mask = None

    plot_utils.apply_plot_params(font_size=12)

    fig = plt.figure()

    years = None
    lats = None
    the_mask = None
    for label, folder in label_to_hles_dir.items():

        y_to_snfl = {}
        y_to_snfldays = {}

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

                y_to_snfl[year_current[0]] = snfl[0]

        years_ord = sorted(y_to_snfl)

        label_to_y_to_snfl[label] = y_to_snfl


        if years is None:
            years = years_ord

        data = np.ma.array([y_to_snfl[y] for y in years_ord])

        if the_mask is None:
            the_mask = data[0].mask

        solver = Eof(data)


        # eof = solver.eofsAsCorrelation(neofs=4)
        eof = solver.eofs(neofs=4)

        pc = solver.pcs(pcscaling=0)
        label_to_varfraction[label] = solver.varianceFraction()

        label_to_pc[label] = pc

        label_to_eof[label] = eof



        # save data for Diro
        print(pc.shape)
        df = pd.DataFrame(data=pc, index=years_ord)
        df.to_csv("{}_pc.csv".format(label))


        plt.plot(years_ord, pc[0, :].copy(), label_to_line_style[label], linewidth=2,
                 label=label)

    plt.legend(loc="upper right")

    plt.ylabel(units)
    plt.xlabel("Year")
    plt.xticks(years)

    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.savefig(str(label_to_hles_dir["Obs"].joinpath("pc{}_{}.png".format(npc, vname))), bbox_inches="tight")

    plt.close(fig)


    # plot the eofs


    plot_utils.apply_plot_params(font_size=12, width_cm=20, height_cm=8)


    for eof_ind in range(3):
        col = 0

        fig = plt.figure()
        gs = GridSpec(1, len(label_to_eof))

        for label, eof_field in label_to_eof.items():

            ax = fig.add_subplot(gs[0, col])
            to_plot = eof_field[eof_ind].T.copy()
            im = plt.pcolormesh(to_plot, cmap=cm.get_cmap("bwr", 10), vmin=-0.025, vmax=0.025)
            plt.colorbar(im, extend="both")
            ax.set_title("{} (explains {:.2f}$\sigma^2$)".format(label, label_to_varfraction[label][eof_ind]))

            col += 1

        fig.tight_layout()
        plt.savefig(str(label_to_hles_dir["Obs"].joinpath("eof_raw_{}_{}.png".format(eof_ind + 1, vname))), bbox_inches="tight")
        plt.close(fig)


if __name__ == '__main__':
    main()