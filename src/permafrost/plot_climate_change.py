from netCDF4 import Dataset
import os
from matplotlib import gridspec
import matplotlib
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import my_colormaps
import draw_regions
from active_layer_thickness import CRCMDataManager
import matplotlib.pyplot as plt
import compare_mean_alt_and_from_climatology as alt_mod
from rpn import level_kinds
from util import plot_utils


__author__ = 'huziy'

import numpy as np



def get_delta(varname, start_year_c, end_year_c, start_year_f, end_year_f, data_folder = "",
              level = -1, level_kind = -1, months = None, percentage = False):
    """
    varname = I5 - > SWE
    varname = TT - > 2m temperature
    """
    dm = CRCMDataManager(data_folder=data_folder)
    data_c = dm.get_mean_over_months_of_2d_var(start_year_c, end_year_c, months = months, var_name=varname,
        level=level, level_kind = level_kind)
    data_f = dm.get_mean_over_months_of_2d_var(start_year_f, end_year_f, months = months, var_name=varname,
        level=level, level_kind = level_kind)
    if percentage:
        return (data_f - data_c)/data_c * 100
    return data_f - data_c


def plot_column(names = None, name_to_data = None, title = "", x = None,y=None,
                basemap = None,mask = None, img_file = "", cmap = None, vminmax = None, units = None, extend = "both"):
    fig = plt.figure()
    gs = gridspec.GridSpec(len(names),2, width_ratios=[1,0.025])
    all_axes = []
    all_img = []
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[i,0])
        assert isinstance(ax, Axes)
        hc = np.ma.masked_where(mask, name_to_data[name])
        img = basemap.pcolormesh(x, y, hc, cmap = cmap, vmax = vminmax[1],vmin = vminmax[0])
        if not i:
            ax.set_title(title)
        i += 1
        ax.set_ylabel("CRCM ({0})".format(name))
        all_axes.append(ax)
        all_img.append(img)



    i = 0
    axs_to_hide = []
    #zones and coastlines
    for the_ax, the_img in zip(all_axes, all_img):
        #divider = make_axes_locatable(the_ax)
        #cax = divider.append_axes("right", "5%", pad="3%")
        #cb = fig.colorbar(the_img,  cax = cax, extend = extend)
        #cax.set_title("{0} \n".format(units))
        assert isinstance(the_ax, Axes)
        basemap.drawcoastlines(ax = the_ax, linewidth=0.5)
        shape_info = basemap.readshapefile("data/pf_4/permafrost8_wgs84/permaice", name="zone",
                ax=the_ax, linewidth=1.5, drawbounds=False)


        for nshape,seg in enumerate(basemap.zone):
            if basemap.zone_info[nshape]["EXTENT"] != "C": continue
            poly = matplotlib.patches.Polygon(seg,edgecolor = "k", facecolor="none", zorder = 10, lw = 1.5)
            the_ax.add_patch(poly)


    cax = fig.add_subplot(gs[:,1])
    cb = fig.colorbar(all_img[0],  cax = cax, extend = extend)
    cax.set_title("{0} \n".format(units))


    fig.tight_layout(h_pad=0, w_pad=0)

    for the_ax in axs_to_hide:
        the_ax.set_visible(False)

    fig.savefig(img_file)




    pass

def plot_climate_change():

    start_year_current = 1981
    end_year_current = 2010

    start_year_future = 2071
    end_year_future = 2100

    periods = "F2 - C"

    #plot changes
    sim_data_folder = "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/era40_driven_b1"
    coord_file = os.path.join(sim_data_folder, "pmNorthAmerica_0.44deg_ERA40-Int_B1_200812_moyenne")
    basemap, lons2d, lats2d = draw_regions.get_basemap_and_coords(resolution="c",
        file_path = coord_file, llcrnrlat=40.0, llcrnrlon=-145, urcrnrlon=-20, urcrnrlat=74, anchor="W"
    )
    assert isinstance(basemap, Basemap)

    #basemap.transform_scalar()

    #basemap = Basemap()
    lons2d[lons2d > 180] -= 360

    x, y = basemap(lons2d, lats2d)


    cmap = my_colormaps.get_red_blue_colormap(ncolors=10, reversed=True)

    sim_names = ["MPI",
        "CanESM"
    ]
    simname_to_path = {
        "MPI": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_MPI_B1",
        "CanESM": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1"


    }
    permafrost_mask = draw_regions.get_permafrost_mask(lons2d, lats2d)
    mask_cond = (permafrost_mask <= 0) | (permafrost_mask >= 2)



    #alt
#    folder = "/home/huziy/skynet1_rech3/cordex/for_Samira"
#    alt_sim_name_to_path = {
#        "MPI": os.path.join(folder, "alt_mpi_b1_yearly.nc"),
#        "CanESM": os.path.join(folder, "alt_canesm_b1_yearly.nc")
#    }
    alt_sim_name_to_path = simname_to_path


    name_to_delta = {}
    for name in sim_names:
        path = alt_sim_name_to_path[name]

        #select data and needed alt
        #ds = Dataset(path)
        #years = ds.variables["year"][:]
        #hct = ds.variables["alt"][(years >= start_year_current) & (years <= end_year_current),:,:]

        #print "hct.shape = ", hct.shape
        #hc = alt_mod.get_alt_using_nyear_rule(hct, nyears = 2)

        #hft = ds.variables["alt"][(years >= start_year_future) & (years <= end_year_future),:,:]
        #hf = alt_mod.get_alt_using_nyear_rule(hft, nyears = 2)

        dm = CRCMDataManager(data_folder=path)
        hc = dm.get_alt_using_monthly_mean_climatology(xrange(start_year_current,end_year_current+1))
        hf = dm.get_alt_using_monthly_mean_climatology(xrange(start_year_future,end_year_future+1))

        good_points = (hf >= 0) & (hc > 0) & (~mask_cond)
        d = np.ma.masked_all(hf.shape)
        d[good_points] = (hf[good_points] - hc[good_points]) / hc[good_points] * 100.0
        name_to_delta[name] = d

        d = (hf - hc) / hc * 100.0
        print name
        the_is, the_js = np.where((hf < hc) & good_points)

#        for i,j in zip(the_is, the_js):
#            print(i,j,lons2d[i,j], lats2d[i,j], hc[i,j], hf[i,j], d[i,j])
#        if name == "CanESM":
#            raise Exception

    plot_column(names=sim_names, name_to_data=name_to_delta,title="ALT, {0}".format(periods), x = x, y = y, basemap=basemap,
        img_file="alt_b1_cc1.png", cmap=cmap, vminmax=(-100, 100), mask=mask_cond, units = "%", extend = "both"
    )


    #swe
    cmap_swe = my_colormaps.get_red_blue_colormap(ncolors=10, reversed=False)

    name_to_delta = {}
    for name in sim_names:
        path = simname_to_path[name]
        name_to_delta[name] = get_delta("PR", start_year_current, end_year_current,
            start_year_future, end_year_future, data_folder = path, months=[6,7,8], percentage=True)

    plot_column(names=sim_names, name_to_data=name_to_delta,title="PREC, {0}, JJA".format(periods), x = x, y = y, basemap=basemap,
        img_file="pr_b1_cc1.png", cmap=cmap_swe, vminmax=(-100, 100), mask=mask_cond,
        units="%", extend = "both"
    )



    #temperature
    sim_names = ["MPI",
                 "CanESM"]
    simname_to_path = {
         "MPI": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_MPI_B1_dm",
         "CanESM": "/home/huziy/skynet1_rech3/cordex/CORDEX_DIAG/NorthAmerica_0.44deg_CanESM_B1_dm"
    }

    name_to_delta = {}
    for name in sim_names:
        path = simname_to_path[name]
        name_to_delta[name] = get_delta("TT", start_year_current, end_year_current,
            start_year_future, end_year_future, data_folder = path, months=[6,7,8],
            level=1, level_kind=level_kinds.HYBRID)


    plot_column(names=sim_names, name_to_data=name_to_delta,title="2m Tempreature, {0}, JJA".format(periods), x = x, y = y, basemap=basemap,
        img_file="tt_b1_cc1.png", cmap=my_colormaps.get_red_colormap(ncolors = 8), vminmax=(1, 5), mask=mask_cond, units="$^{\\circ} {\\rm C}$", extend = "max"
    )







def main():
    plot_climate_change()
    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(width_pt=None, width_cm=28,height_cm=30, font_size=25)
    main()
    print "Hello world"
  