import os
from matplotlib import gridspec
from matplotlib.figure import Figure
import my_colormaps
import draw_regions
from rpn.rpn import RPN
from util import plot_utils

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt

def main():

    sim_names = ["MPI", "CanESM" ]
    data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/from_Leo"
    current_paths = [
         os.path.join(data_folder, "c_MPI-ESM-LR_pr_JJA_1981-2010_rg"),
         os.path.join(data_folder, "c_CanESM2_pr_JJA_1981-2010_rg")
    ]

    future_paths = [

         os.path.join(data_folder, "c_MPI-ESM-LR_pr_JJA_2071-2100_rg"),
         os.path.join(data_folder, "c_CanESM2_pr_JJA_2071-2100_rg")

    ]



    fig = plt.figure()
    assert isinstance(fig, Figure)


    gs = gridspec.GridSpec(2,2, width_ratios=[1, 0.07], bottom=0.01, left=0.01, top=0.96,
        hspace=0.1, wspace=0)
    cmap = my_colormaps.get_red_blue_colormap(ncolors=10)

    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords()
    lons2d[lons2d > 180] -= 360

    x, y = basemap(lons2d, lats2d)
    i = 0
    img = None
    for the_name, cur_path, fut_path in zip(sim_names, current_paths, future_paths):

        ax = fig.add_subplot(gs[i,0])

        rc = RPN(cur_path)
        data_c = rc.get_first_record_for_name("PR")

        rf = RPN(fut_path)
        data_f = rf.get_first_record_for_name("PR")

        img = basemap.pcolormesh(x, y, (data_f - data_c) / data_c * 100.0,
            ax = ax, cmap = cmap, vmax = 100, vmin = -100)
        basemap.drawcoastlines(ax = ax)
        ax.set_title(the_name)
        i += 1


    ax = fig.add_subplot(gs[:,1])
    ax.set_aspect(40)
    fig.colorbar(img, cax = ax, extend = "both")

    #fig.tight_layout()

    #plt.show()
    fig.savefig("clim_change_pr_mpi_canesm.png")


    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(width_pt=None, width_cm=23,height_cm=40, font_size=25)
    main()
    print "Hello world"
  