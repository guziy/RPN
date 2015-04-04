import itertools
from mpl_toolkits.basemap import Basemap, maskoceans
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from rpn.domains.rotated_lat_lon import RotatedLatLon
from scipy.stats import ttest_ind

__author__ = 'huziy'

import matplotlib.pyplot as plt
from rpn.rpn import RPN
import numpy as np

from permafrost.active_layer_thickness import CRCMDataManager


def plot_values(basemap, lons2d, lats2d, data1, label1, data2, label2, depth_to_bedrock):
    fig = plt.figure()

    # max_value = 5.0
    # step = 2

    #clevs = np.arange(0, max_value + step, step).tolist()
    clevs = [0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2.0, 3.0, 4.0, 5.0]
    cmap = cm.get_cmap("jet", len(clevs) - 1)

    norm = BoundaryNorm(clevs, len(clevs) - 1)

    x, y = basemap(lons2d, lats2d)

    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])


    #draw alt1
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(label1)
    im = basemap.pcolormesh(x, y, data1, cmap=cmap, ax=ax, norm=norm, vmin=clevs[0], vmax=clevs[-1])
    basemap.drawcoastlines(ax=ax)
    #fill background with grey
    basemap.drawmapboundary(fill_color="0.75")

    #draw alt2
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title(label2)
    im = basemap.pcolormesh(x, y, data2, cmap=cmap, ax=ax, norm=norm, vmin=clevs[0], vmax=clevs[-1])
    basemap.drawcoastlines(ax=ax)

    #fill background with grey
    basemap.drawmapboundary(fill_color="0.75")

    cax = fig.add_subplot(gs[0, 2])
    plt.colorbar(im, cax=cax)
    fig.savefig("alts_for_diff_dpth.png")


def plot_differences(basemap, lons2d, lats2d, depth_to_bedrock, delta, label="Fixed dpth - Real dpth",
                     pvalue=None, swe_diff=None):
    fig = plt.figure(figsize=(18, 6))

    max_delta = 1.0
    step = 0.2

    clevs = np.arange(-max_delta, 0, step).tolist()
    clevs += [0, ] + np.arange(step, max_delta + step, step).tolist()
    cmap = cm.get_cmap("seismic", len(clevs) - 1)

    norm = BoundaryNorm(clevs, len(clevs) - 1)

    x, y = basemap(lons2d, lats2d)

    gs = GridSpec(1, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(label)

    im = basemap.pcolormesh(x, y, delta, cmap=cmap, ax=ax, norm=norm, vmin=clevs[0], vmax=clevs[-1])
    basemap.drawcoastlines(ax=ax, linewidth=0.5)
    basemap.colorbar(im)
    basemap.drawmapboundary(fill_color="0.75")

    if pvalue is not None:
        cs = basemap.contourf(x, y, pvalue, levels=[0.05, 1], colors="none", hatches=["\\\\", ])
        # create a legend for the contour set
        artists, labels = cs.legend_elements()
        plt.legend(artists, labels, handleheight=2)



    # Plot depth to bedrock
    print("plotting depth to bedrock")
    clevs = [0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5]
    cmap = cm.get_cmap("jet", len(clevs) - 1)
    norm = BoundaryNorm(clevs, len(clevs) - 1)
    ax = fig.add_subplot(gs[0, 1])
    im = basemap.pcolormesh(x, y, depth_to_bedrock,
                            cmap=cmap, ax=ax, norm=norm, vmin=clevs[0], vmax=clevs[-1])
    basemap.drawcoastlines(ax=ax, linewidth=0.5)
    basemap.colorbar(im, ax=ax)
    basemap.drawmapboundary(fill_color="0.75")
    ax.set_title("DPTH(~, m)")


    #Plot swe
    print("plotting swe differences")
    clevs = [0.5, 1, 2, 3, 5, 10]
    clevs = [-lev for lev in reversed(clevs)] + [0, ] + clevs
    cmap = cm.get_cmap("seismic", len(clevs) - 1)
    norm = BoundaryNorm(clevs, len(clevs) - 1)

    ax = fig.add_subplot(gs[0, 2])
    im = basemap.pcolormesh(x, y, swe_diff,
                            cmap=cmap, ax=ax, norm=norm, vmax=clevs[-1], vmin=clevs[0])
    basemap.drawcoastlines(ax=ax)
    basemap.colorbar(im, ax=ax)
    basemap.drawmapboundary(fill_color="0.75")
    ax.set_title(r"$\Delta {\rm SWE}$ (Winter, mm)")



    # cs = basemap.contour(x, y, 3.6 - depth_to_bedrock, ax=ax, colors="k",
    #                     levels=[0, 0.1, 0.5, 1, 2, 3.6, 4], linewidth=0.02)
    #plt.clabel(cs, inline=1, fontsize=9)



    fig.savefig("alt_change_due_to_dpth.png", bbox_inches="tight")


def main():
    soiltemp_var_name = "TBAR"
    path1 = "/skynet1_rech3/huziy/class_offline_simulations_VG/dpth_var/CLASS_output_CLASSoff1_Arctic_0.5_ERA40_dpth_to_bdrck_var_1980-2009/TBAR.rpn"
    label1 = "Variable (real) depth to bedrock"

    path2 = "/skynet1_rech3/huziy/class_offline_simulations_VG/dpth_3.6m/CLASS_output_CLASSoff1_Arctic_0.5_ERA40_dpth_to_bdrck_constant_1980-2009/TBAR.rpn"
    label2 = "Fixed depth to bedrock (3.6 m)"

    path_to_dpth_to_bedrock = "/skynet1_rech3/huziy/CLASS_offline_VG/GEOPHYSICAL_FIELDS/test_analysis.rpn"

    # read depth to bedrock
    r = RPN(path_to_dpth_to_bedrock)
    dpth = r.get_first_record_for_name("DPTH")
    r.close()

    layer_widths = [0.1, 0.2, 0.3, 0.5, 0.9, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
                    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]

    print(len(layer_widths))

    crcm_data_manager = CRCMDataManager(layer_widths=layer_widths)

    # Read in the temperature profiles
    r1 = RPN(path1)
    soiltemp1 = r1.get_4d_field(name=soiltemp_var_name)
    lons2d, lats2d = r1.get_longitudes_and_latitudes_for_the_last_read_rec()
    rll = RotatedLatLon(**r1.get_proj_parameters_for_the_last_read_rec())
    b = rll.get_basemap_object_for_lons_lats(lons2d=lons2d, lats2d=lats2d, resolution="c")
    r1.close()

    r2 = RPN(path2)
    soiltemp2 = r2.get_4d_field(name=soiltemp_var_name)
    r2.close()

    dates_sorted = list(sorted(soiltemp1.keys()))
    levels_sorted = list(sorted(list(soiltemp1.items())[0][1].keys()))

    # group dates for each year
    alt1_list = []
    alt2_list = []
    for year, dates_group in itertools.groupby(dates_sorted, lambda par: par.year):
        t1 = []
        t2 = []
        for d in dates_group:
            t1.append([soiltemp1[d][lev] for lev in levels_sorted])
            t2.append([soiltemp2[d][lev] for lev in levels_sorted])

        # Get maximum temperature profiles
        t1max = np.max(t1, axis=0).transpose(1, 2, 0)
        t2max = np.max(t2, axis=0).transpose(1, 2, 0)

        print(t1max.shape, t2max.shape)

        #calculate and save alts
        h1 = crcm_data_manager.get_alt(t1max)
        h2 = crcm_data_manager.get_alt(t2max)

        h1[h1 < 0] = np.nan
        h2[h2 < 0] = np.nan

        alt1_list.append(h1)
        alt2_list.append(h2)

    #take into account permafrost
    alt1_list_pf = []
    alt2_list_pf = []
    n_years_for_pf = 3
    for i in range(len(alt1_list) - n_years_for_pf):
        alt1_list_pf.append(np.max(alt1_list[i:i+n_years_for_pf], axis=0))
        alt2_list_pf.append(np.max(alt2_list[i:i+n_years_for_pf], axis=0))



    # calculate climatological mean
    alt1 = np.mean(alt1_list_pf, axis=0)
    alt2 = np.mean(alt2_list_pf, axis=0)

    print(np.isnan(alt1).any(), np.isnan(alt2).any())

    #mask nans
    alt1 = np.ma.masked_where(np.isnan(alt1), alt1)
    alt2 = np.ma.masked_where(np.isnan(alt2), alt2)

    #calculate change due to fixed depth to bedrock
    delta = alt2 - alt1

    #
    for i in range(len(alt1_list)):
        alt1_list[i] = np.ma.masked_where(np.isnan(alt1_list[i]), alt1_list[i])
        alt2_list[i] = np.ma.masked_where(np.isnan(alt2_list[i]), alt2_list[i])

    #tval, pval = ttest_ind(alt1_list, alt2_list)

    #mask oceans
    lons2d[lons2d > 180] -= 360
    dpth = maskoceans(lons2d, lats2d, dpth)
    delta = np.ma.masked_where(dpth.mask, delta)


    #calculate differences in SWE
    path_swe_1 = path1.replace("TBAR.rpn", "SNO.rpn")
    r1 = RPN(path_swe_1)
    swe1 = r1.get_all_time_records_for_name("SNO")
    r1.close()
    swe1_winter_clim = np.mean(
        [field for key, field in swe1.items() if key.month in [1, 2, 12]], axis=0)
    swe1_winter_clim = np.ma.masked_where((swe1_winter_clim >= 999) | dpth.mask, swe1_winter_clim)

    path_swe_2 = path2.replace("TBAR.rpn", "SNO.rpn")
    r2 = RPN(path_swe_2)
    swe2 = r2.get_all_time_records_for_name("SNO")
    r2.close()
    swe2_winter_clim = np.mean(
        [field for key, field in swe2.items() if key.month in [1, 2, 12]], axis=0)
    swe2_winter_clim = np.ma.masked_where((swe2_winter_clim >= 999) | dpth.mask, swe2_winter_clim)




    #plotting
    print("Start plotting ...")
    plot_differences(b, lons2d, lats2d, dpth, delta, label="ALT(DPTH=3.6m) - ALT(DPTH~)", pvalue=None,
                     swe_diff=swe2_winter_clim - swe1_winter_clim)
    #plot_values(b, lons2d, lats2d, alt1, "ALT(DPTH=3.6m)", alt2, "ALT(DPTH~)", dpth)


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    main()
