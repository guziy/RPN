import os
from matplotlib import cm
from matplotlib.colors import BoundaryNorm

__author__ = 'huziy'

import matplotlib.pyplot as plt
from nemo import nemo_commons
from netCDF4 import Dataset
import numpy as np
import my_colormaps



def main(exp_path="/home/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK/test_fwb_my"):

    data_path = os.path.join(exp_path, "bathy_meter.nc")
    exp_name = os.path.basename(exp_path)

    img_folder = os.path.join("nemo", exp_name)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    img_path = os.path.join(img_folder, "bathy_meter.svg")

    levels = [0, 20, 30, 60, 80, 160, 200, 280, 320, 360, 400]

    b, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(path=data_path, resolution = "i")

    plt.figure(figsize = (16, 8))
    data = Dataset(data_path).variables["Bathymetry"][:]
    x, y = b(lons, lats)
    data = np.ma.masked_where(data < 0.001, data)

    cmap = cm.get_cmap("jet", len(levels) - 1)
    #cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/cosam.rgb", ncolors = len(levels) - 1)
    bn = BoundaryNorm(levels, len(levels) - 1)
    img = b.pcolormesh(x, y, data, cmap = cmap, norm = bn)
    cb = b.colorbar(img, ticks = levels)
    cb.ax.tick_params(labelsize = 30)
    cb.ax.set_title("m", fontsize = 30)
    b.drawcoastlines()
    b.drawrivers()

    b.drawmeridians(np.arange(-140, -50, 5), labels = [0, 0, 0, 1], fontsize = 30)
    b.drawparallels(np.arange(0, 90, 5), labels = [1, 0, 0, 0], fontsize = 30)
    plt.tight_layout(rect=[0.07, 0.0, 1, 1])
    plt.savefig(img_path, dpi = 400)


    plt.show()

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()

    main()