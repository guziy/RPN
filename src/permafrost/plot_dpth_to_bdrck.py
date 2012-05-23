import os
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import draw_regions
import my_colormaps
from rpn.rpn import RPN
from util import plot_utils

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt
def main():
    path = "/home/huziy/skynet1_rech3/cordex/NorthAmerica_0.44deg_ERA40-Int_195801_static_data.rpn"
    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1" #for coordinates
    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file,
        llcrnrlat=40.0, llcrnrlon=-143, urcrnrlon=-20, urcrnrlat=74
    )

    #read depth to bedrock field
    rObj = RPN(path)
    dpth_to_bdrck = rObj.get_first_record_for_name("8L")
    rObj.close()

    #dpth_to_bdrck[:,:] = 4
    bounds = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.6]
    cmap = my_colormaps.get_lighter_jet_cmap(ncolors=len(bounds) - 1)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N)
    x,y = basemap(lons2d, lats2d)
    fig = plt.figure()
    CS = basemap.pcolormesh(x, y, dpth_to_bdrck, norm = norm, cmap = cmap)
    print( dpth_to_bdrck.min(), dpth_to_bdrck.max())
    basemap.drawcoastlines()
    basemap.drawparallels(np.arange(-80.,81.,20.))
    basemap.drawmeridians(np.arange(-180.,181.,20.))

    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="3%")
    cax.set_title("m")
    fig.colorbar(CS, cax = cax, ticks=bounds)




    fig.tight_layout()
    fig.savefig("dpth_to_bdrck.png")





    plt.show()


    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(width_pt=None, width_cm=28,height_cm=30, font_size=25)
    main()
    print "Hello world"
  