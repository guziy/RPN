from netCDF4 import Dataset
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import draw_regions
import my_colormaps
from util import plot_utils

__author__ = 'huziy'

import numpy as np
import os
from active_layer_thickness import CRCMDataManager
import matplotlib.pyplot as plt

def main():
    start_year = 1981
    end_year = 2008

    #mean alt
    path_to_yearly = "alt_era_b1_yearly.nc"
    ds = Dataset(path_to_yearly)

    hm = ds.variables["alt"][:]
    years = ds.variables["year"][:]
    years_sel = np.where(( start_year <= years ) & (years <= end_year))[0]
    hm = hm[np.array(years_sel),:,:]

    good_points = ~np.any(hm < 0, axis = 0)

    hm2d = np.ma.masked_all(good_points.shape)


    hm2d[good_points] = np.mean( hm[ : , good_points],
                        axis = 0)
    ds.close()


    #alt from climatology
    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"
    dm = CRCMDataManager(data_folder=sim_data_folder)
    hc = dm.get_alt_using_monthly_mean_climatology(xrange(start_year,end_year+1))

    print hc.min(),hc.max()


    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-10
    )

    x, y = basemap(lons2d, lats2d)
    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 3)

    plot_utils.apply_plot_params(width_pt=None, width_cm=25,height_cm=35, font_size=12)
    fig = plt.figure()
    assert isinstance(fig, Figure)


    h_max = 10
    clevels = np.arange(0,h_max+1,1)
    gs = gridspec.GridSpec(3,1)

    all_axes = []
    all_img = []


    ax = fig.add_subplot(gs[0,0])
    hm2d = np.ma.masked_where(mask_cond | (hm2d > h_max), hm2d)
    img = basemap.contourf(x, y, hm2d, levels = clevels)
    ax.set_title("Mean ALT")
    all_axes.append(ax)
    all_img.append(img)
    print("hm2d(min,max) = ",hm2d.min(), hm2d.max())

    ax = fig.add_subplot(gs[1,0])
    hc = np.ma.masked_where(hc < 0, hc)
    hc = np.ma.masked_where(mask_cond | (hc > h_max) | hm2d.mask, hc)
    img = basemap.contourf(x, y, hc, levels = clevels)
    all_img.append(img)
    all_axes.append(ax)
    ax.set_title("ALT from climatology")
    print("hc(min,max) = ",hc.min(), hc.max())


    ax = fig.add_subplot(gs[2,0])
    delta = hm2d - hc
    delta = np.ma.masked_where(hc.mask | hm2d.mask, delta)
    img = basemap.contourf(x, y, delta, levels = np.arange(-1,1.2,0.2),ax = ax,
        cmap = my_colormaps.get_red_blue_colormap())
    all_img.append(img)
    all_axes.append(ax)
    ax.set_title("Mean - Derived from climatology")

    #print(10 * "--")
    #print(hm[:,(hm2d < hc) & ~(hc.mask | hm2d.mask)][:,5])
    #print(10 * "--")
    #print(hc[(hm2d < hc) & ~(hc.mask | hm2d.mask)][5])

    print(np.where((hm2d < hc) & ~(hc.mask | hm2d.mask)))

    for the_ax, the_img in zip( all_axes, all_img ):
        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=0.5)
        divider = make_axes_locatable(the_ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = fig.colorbar(the_img,  cax = cax)
        CS = basemap.contour(x,y, permafrost_mask, levels = [1,2,3],
            ax = the_ax, colors = "k", linewidth= 5)
        #the_ax.clabel(CS,colors = 'k', fmt="%d" , fontsize=8)


    fig.tight_layout()
    #cax_to_hide.set_visible(False)
    fig.savefig("alt_b1.png")


    plt.figure()
    plt.contourf(np.std(hm, axis=0).transpose())
    plt.show()
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  