import os
from matplotlib.colors import BoundaryNorm
from nemo import nemo_commons
from nemo.nemo_output_manager import NemoOutputManager

__author__ = 'huziy'


import matplotlib.pyplot as plt
import numpy as np


def main_for_lake(bathy_path = "",
                  basemap = None,
                  manager_u = None, manager_v=None,
                  scalar_manager=None, img_dir=""):

    """

    :param bathy_path: file used for land sea mask
    :param basemap:
    :param lons:
    :param lats:
    :param manager_u:
    :param manager_v:
    :return:
    """
    the_mask = nemo_commons.get_mask(bathy_path)
    print the_mask.shape
    scalar_levels = np.arange(-25, 30, 5)



    for frame in range(manager_u.get_total_number_of_time_frames()):

        fig = plt.figure()


        #bounds = [0, 0.02, 0.05, 0.08, 0.09, 0.1]
        bounds = np.arange(-20, 21, 1)
        bn = BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)
        #img = plt.pcolormesh(data, vmin = bounds[0], vmax = bounds[-1], cmap = cm.get_cmap("jet", len(bounds) - 1), norm = bn)

        b_local, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(path=bathy_path, resolution="i")
        x, y = basemap(lons, lats)


        if scalar_manager is not None:
            data = scalar_manager.get_next_time_frame_data()
            print data.shape
            img = basemap.pcolormesh(x, y, np.ma.masked_where(~the_mask, data),
                                     vmin=scalar_levels[0], vmax=scalar_levels[-1], zorder=-6)
            basemap.colorbar()

        u, v = manager_u.get_next_time_frame_data(), manager_v.get_next_time_frame_data()
        u = np.ma.masked_where(~the_mask, u)
        v = np.ma.masked_where(~the_mask, v)
        # qp = basemap.quiver(x, y, u, v, scale=1.5)
        c = np.sqrt(u ** 2 + v ** 2)
        qp = basemap.streamplot(x, y, u, v, color="k", density=(5, 5), linewidth=3 * c / c.max())
        basemap.drawcoastlines()
        basemap.drawmapboundary(fill_color="gray")
        plt.title(str(manager_u.get_current_time()).split()[0])
        print str(manager_u.get_current_time()).split()[0]

        fig.savefig(os.path.join(img_dir, "{0:08d}.png".format(frame)))
        plt.close(fig)



def main():

    whole_domain_bathymetry = \
        "/skynet3_rech1/huziy/NEMO_OFFICIAL/Simulations/1981-2000_Sim_per_lake_100yr_spinup_LIM3/Huron/bathy_meter.nc"

    basemap, lons, lats = nemo_commons.get_basemap_and_coordinates_from_file(path=whole_domain_bathymetry,
                                                                             resolution="h")



    data_folder = "/home/huziy/skynet3_rech1/NEMO_OFFICIAL/Simulations/1981-2000_Sim_per_lake_100yr_spinup_LIM3/Huron"
    manager_u = NemoOutputManager(
        file_path = os.path.join(data_folder, "GLK_10d_19810101_20001231_grid_U.nc"),
        var_name="vozocrtx")

    manager_v = NemoOutputManager(
        file_path=os.path.join(data_folder, "GLK_10d_19810101_20001231_grid_V.nc"),
        var_name="vomecrty")


    scalar_manager = NemoOutputManager(
        file_path=os.path.join(data_folder, "GLK_10d_19810101_20001231_grid_T.nc"),
        var_name="sosstsst")

    img_dir = "animations_nemo/Huron"

    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    main_for_lake(
        bathy_path=os.path.join(data_folder, "bathy_meter.nc"),
        manager_u=manager_u, manager_v=manager_v,
        basemap=basemap, scalar_manager = scalar_manager, img_dir=img_dir)



if __name__ == "__main__":
    main()