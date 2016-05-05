

import os
import numpy as np
from rpn import level_kinds
from rpn.rpn_multi import MultiRPN
import matplotlib.pyplot as plt

from domains.rotated_lat_lon import RotatedLatLon


def main():
    #path = "/RECH/data/Simulations/CRCM5/North_America/NorthAmerica_0.44deg_ERA40-Int_B1/Diagnostics/NorthAmerica_0.44deg_ERA40-Int_B1_2007{:02d}"
    path = "/RESCUE/skynet3_rech1/huziy/from_guillimin/new_outputs/current_climate_30_yr_sims/quebec_0.1_crcm5-hcd-rl-intfl_ITFS/Samples/quebec_crcm5-hcd-rl-intfl_1988{:02d}"

    months = [6, 7, 8]

    pm_list = []
    dm_list = []
    for m in months:
        print(path.format(m))

        month_folder = path.format(m)
        for fn in os.listdir(month_folder):

            # if not fn.endswith("moyenne"):
            #    continue

            if fn.startswith("pm"):
                pm_list.append(os.path.join(month_folder, fn))
            elif fn.startswith("dm"):
                dm_list.append(os.path.join(month_folder, fn))



    pm = MultiRPN(pm_list)
    dm = MultiRPN(dm_list)

    tsurf_mean = np.mean([field for field in pm.get_all_time_records_for_name_and_level(varname="J8").values()], axis=0)
    tair_mean = np.mean([field for field in dm.get_all_time_records_for_name_and_level(varname="TT", level=1, level_kind=level_kinds.HYBRID).values()], axis=0)


    lons, lats = pm.get_longitudes_and_latitudes_of_the_last_read_rec()

    projparams = pm.linked_robj_list[0].get_proj_parameters_for_the_last_read_rec()

    rll = RotatedLatLon(**projparams)
    bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)

    xx, yy = bmp(lons, lats)


    plt.figure()
    cs = bmp.contourf(xx, yy, tsurf_mean - 273.15, 40)
    bmp.drawcoastlines()
    plt.title("Tsurf")
    plt.colorbar()

    plt.figure()
    bmp.contourf(xx, yy, tair_mean, levels=cs.levels, norm=cs.norm, cmap=cs.cmap)
    bmp.drawcoastlines()
    plt.title("Tair")
    plt.colorbar()

    plt.figure()
    bmp.contourf(xx, yy, tsurf_mean - 273.15 - tair_mean, levels=np.arange(-2, 2.2, 0.2), cmap=cs.cmap)
    bmp.drawcoastlines()
    plt.title("Tsurf - Tair")
    plt.colorbar()



    pm.close()
    dm.close()

    plt.show()

if __name__ == '__main__':
    main()