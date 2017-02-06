from collections import OrderedDict
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter, LogFormatter, LogLocator, FuncFormatter, LinearLocator

from application_properties import main_decorator
from crcm5.mh_domains.show_domain_with_drainage_area import show_domain

from hydrosheds.plot_directions import plot_directions
from crcm5.mh_domains import default_domains
from util import plot_utils

import matplotlib.pyplot as plt
import numpy as np


img_folder = Path("mh").joinpath("engage_report")


FACC = "accumulation_area"
LAKE_FRACTION = "lake_fraction"


vname_to_clevs = {
    FACC: [10 ** p for p in range(1, 8)],
    LAKE_FRACTION: np.arange(0, 1.1, 0.1)
}


vname_to_units = {
    FACC: r"km$^2$",
    LAKE_FRACTION: ""

}

vname_to_cbar_ticks = {
    FACC: LogLocator(),
    LAKE_FRACTION: LinearLocator(),
}



def format_ticks_ten_pow(x, pos):
    return r"10$^{{{:.0f}}}$".format(np.log10(x))

vname_to_cbar_format = {
    FACC: FuncFormatter(format_ticks_ten_pow),
    LAKE_FRACTION: None
}


@main_decorator
def main():
    # directions_file_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_bc-mh_0.44deg.nc"
    # plot_directions(nc_path_to_directions=directions_file_path,
    #                 grid_config=default_domains.bc_mh_044,
    #                 shape_path_to_focus_polygons=default_domains.MH_BASINS_PATH)
    #



    # vname = "accumulation_area"
    vname = LAKE_FRACTION
    basin_border_width = 0.5

    plot_utils.apply_plot_params(width_cm=25, height_cm=10, font_size=8)

    img_folder_path = Path(img_folder)

    fig1 = plt.figure()
    gs = GridSpec(2, 3, wspace=0.1, hspace=0.0, height_ratios=[1, 0.04])


    dir_file_to_domain_config =  OrderedDict([
        ("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_mh_0.44deg.nc", default_domains.bc_mh_044),
        ("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_mh_0.22deg.nc", default_domains.bc_mh_022),
        ("/RESCUE/skynet3_rech1/huziy/directions_for_ManitobaHydro/directions_mh_0.11deg.nc", default_domains.bc_mh_011)
    ])

    for col, (dir_file, domain_config) in enumerate(dir_file_to_domain_config.items()):

        ax = fig1.add_subplot(gs[0, col])

        print("Processing {} for {}".format(dir_file, domain_config))

        im = show_domain(domain_config, draw_rivers=True, ax=ax,
                    path_to_shape_with_focus_polygons=default_domains.MH_BASINS_PATH,
                    include_buffer=False,
                    directions_file=dir_file,
                    clevels=vname_to_clevs[vname],
                    draw_colorbar=False,
                    basin_border_width=basin_border_width,
                    nc_varname_to_show=vname)


    cbax = fig1.add_subplot(gs[1,:])
    cbax.set_aspect(1.0 / 20.0)
    plt.colorbar(im, cax=cbax, orientation="horizontal", ticks=vname_to_cbar_ticks[vname], format=vname_to_cbar_format[vname])
    cbax.set_xlabel(vname_to_units[vname])


    if not img_folder.exists():
        img_folder.mkdir()

    # fig1.tight_layout()
    fig1.savefig(str(img_folder_path.joinpath("bc_mh_011_022_044_{}.png".format(vname))), bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig1)





if __name__ == '__main__':
    main()