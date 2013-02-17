import os
from matplotlib import cm
import matplotlib
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import maskoceans
from crcm5.model_data import Crcm5ModelDataManager
import my_colormaps
from rpn.rpn import RPN
from util import plot_utils

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt

def main():
    folder = "/home/huziy/skynet3_rech1/geof_lake_infl_exp"
    fName = "geophys_Quebec_0.1deg_260x260_with_dd_v6"
    path = os.path.join(folder, fName)

    rObj = RPN(path)






    mg = rObj.get_first_record_for_name("MG")
    #j2 = rObj.get_first_record_for_name("J2")

    levs = [0,100,200,300,500,700, 1000, 1500,2000, 2800]
    norm = BoundaryNorm(levs, len(levs) - 1)

    me = rObj.get_first_record_for_name("ME")
    lons2d, lats2d = rObj.get_longitudes_and_latitudes_for_the_last_read_rec()

    lons2d[lons2d > 180] -= 360
    #me_to_plot = np.ma.masked_where(mg < 0.4, me)
    me_to_plot = me
    #print me_to_plot.min(), me_to_plot.max()

    basemap = Crcm5ModelDataManager.get_omerc_basemap_using_lons_lats(lons2d=lons2d,
        lats2d=lats2d, resolution="l")
    x, y = basemap(lons2d, lats2d)

    plt.figure( )
    ax = plt.gca()
    #the_cmap = cm.get_cmap(name = "gist_earth", lut=len(levs) -1)
    #the_cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/topo_15lev.rgb")
    the_cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/OceanLakeLandSnow.rgb",
                                                ncolors=  len(levs) - 1)


    #new_cm = matplotlib.colors.LinearSegmentedColormap('colormap',new_dict,len(levs) - 1)

    me_to_plot = maskoceans(lons2d, lats2d, me_to_plot, resolution="l")
    basemap.contourf(x, y, me_to_plot,
        cmap=the_cmap, levels = levs, norm = norm)



    #basemap.fillcontinents(color = "none", lake_color="aqua")
    basemap.drawmapboundary(fill_color='#479BF9')
    basemap.drawcoastlines()
    basemap.drawmeridians(np.arange(-180, 180, 20),labels=[1,0,0,1])
    basemap.drawparallels(np.arange(45, 75, 15), labels=[1,0,0,1])



    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(ticks = levs, cax = cax)

    basemap.scatter(x,y, color="k", s = 1, linewidths = 0, ax = ax, zorder=2)

    margin = 20
    x1 = x[margin, margin]
    x2 = x[-margin, margin]
    y1 = y[margin, margin]
    y2 = y[margin, -margin]
    pol_corners = ((x1,y1),(x2,y1),(x2,y2),(x1, y2))
    ax.add_patch(Polygon(xy = pol_corners, fc = "none", ls="dashed", lw = 3))

    plt.tight_layout()
    plt.savefig("free_domain_260x260.jpeg")
    rObj.close()



    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(width_cm=15, height_cm=15)
    main()
    print "Hello world"
  