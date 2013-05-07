from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from crcm5.model_data import Crcm5ModelDataManager
from rpn.rpn import RPN

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt


def main():
    path = "/skynet3_rech1/huziy/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup2/Samples/quebec_crcm5-hcd-rl-intfl_197901/pm1979010100_00000000p"

    rObj = RPN(path)


    sani = rObj.get_first_record_for_name("SANI")
    lons, lats = rObj.get_longitudes_and_latitudes_for_the_last_read_rec()

    basemap = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(lons2d=lons, lats2d = lats)
    x, y = basemap(lons, lats)

    sani = np.ma.masked_where(sani < 2, sani)

    levels = [2,3,4,5,6,7,8,10,15,20,25,30,40]
    cmap = cm.get_cmap("jet", len(levels) - 1)
    bn = BoundaryNorm(levels, cmap.N)
    fig = plt.figure()
    basemap.contourf(x, y, sani, levels = levels, cmap = cmap, norm = bn)
    basemap.drawcoastlines()
    basemap.colorbar(ticks = levels)

    fig.tight_layout()
    fig.savefig("soil_anis.jpeg")


    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, height_cm=25, width_cm=39)
    main()
    print "Hello world"
  