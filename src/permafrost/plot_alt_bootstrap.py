import os
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from rpn.domains.rotated_lat_lon import RotatedLatLon
from rpn.rpn import RPN
from active_layer_thickness import CRCMDataManager
import numpy as np

__author__ = 'huziy'

import matplotlib.pyplot as plt


def plot_values(basemap, lons2d, lats2d, data, label):
    fig = plt.figure()

    # max_value = 5.0
    # step = 2

    # clevs = np.arange(0, max_value + step, step).tolist()
    clevs = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4.0, 4.5, 5.0]
    cmap = cm.get_cmap("jet", len(clevs) - 1)

    #norm = BoundaryNorm(clevs, len(clevs) - 1)

    x, y = basemap(lons2d, lats2d)

    gs = GridSpec(1, 1)


    #draw alt1
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title(label)
    #im = basemap.pcolormesh(x, y, data, cmap=cmap, ax=ax, norm=norm, vmin=clevs[0], vmax=clevs[-1])
    im = basemap.contourf(x, y, data, cmap=cmap, ax=ax, levels=clevs, extend="both")
    #cmap.set_over(cmap(clevs[-1]))
   # cmap.set_under(cmap(clevs[0]))

    basemap.colorbar(im)
    basemap.drawcoastlines(ax=ax)
    #fill background with grey
    basemap.drawmapboundary(fill_color="0.75")

    fig.savefig("alts_peat_{}.png".format(label))


def main():
    #data_folder = "/home/huziy/b2_fs2/sim_links_frm_Katja/Arctic_0.5deg_Peat_SC_26L_CanHR85_spn_Vspng_Diagnostics/1971-2000"

    data_folder = "/home/huziy/b2_fs2/sim_links_frm_Katja/Arctic_0.5deg_Peat_SC_26L_CanHR85_spn_Vspng_Diagnostics/2071-2100"
    var_name = "I0"

    # layer_widths = [0.1, 0.2, 0.3, 0.5, 0.9, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    # 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    #                 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]


    layer_widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    1.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, ]

    print len(layer_widths)

    crcm_data_manager = CRCMDataManager(layer_widths=layer_widths, data_folder=data_folder)


    #get the mask
    r = RPN("/b1_fs2/winger/Arctic/land_sea_glacier_lake_mask_free")
    msk = r.get_first_record_for_name("FMSK")

    start_year = 2071
    end_year = 2100

    # alt - using globally max temperature profile
    #alt = crcm_data_manager.get_alt_using_files_in(data_folder, vname=var_name)




    #calculate alt for each year and then take mean
    alt_list = []
    for y in range(start_year, end_year + 1):
        tmax = crcm_data_manager.get_Tmax_profiles_for_year_using_monthly_means(y, var_name=var_name)
        alt1 = crcm_data_manager.get_alt(tmax)
        alt1[alt1 < 0] = np.nan
        alt_list.append(alt1)

    alt = np.mean(alt_list, axis=0)
    alt[np.isnan(alt)] = -1

    alt = np.ma.masked_where(alt < 0, alt)
    alt = np.ma.masked_where(msk < 0.1, alt)

    #get the coordinates
    fcoord = os.listdir(data_folder)[0]
    fcoord = os.path.join(data_folder, fcoord)
    r = RPN(fcoord)
    i0 = r.get_first_record_for_name(var_name)
    lons2d, lats2d = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    rll = RotatedLatLon(**r.get_proj_parameters_for_the_last_read_rec())
    basemap = rll.get_basemap_object_for_lons_lats(lons2d=lons2d, lats2d=lats2d)

    plot_values(basemap, lons2d, lats2d, alt, "{}-{}(Arctic-peat-26L)".format(start_year, end_year))


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    main()
