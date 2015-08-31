import os

from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap
from scipy.spatial.kdtree import KDTree
from matplotlib import cm
import matplotlib.pyplot as plt

from crcm5.model_data import Crcm5ModelDataManager
from domains.rotated_lat_lon import RotatedLatLon
from data.swe import SweDataManager
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np

#compare errors in winter swe

def upscale(manager_in, manager_out, swe_in, nneighbours = 25):
    assert isinstance(manager_in, Crcm5ModelDataManager)
    assert isinstance(manager_out, Crcm5ModelDataManager)

    lons_in_1d = manager_in.lons2D.flatten()
    lats_in_1d = manager_in.lats2D.flatten()

    x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lons_in_1d, lats_in_1d)

    kdtree = KDTree(list(zip(x0,y0,z0)))

    lons_out_1d = manager_out.lons2D.flatten()
    lats_out_1d = manager_out.lats2D.flatten()

    x1, y1, z1 = lat_lon.lon_lat_to_cartesian(lons_out_1d, lats_out_1d)
    dd, ii = kdtree.query(list(zip(x1, y1, z1)), k=nneighbours)

    print(ii.shape)
    swe_in_1d = swe_in.flatten()

    return np.mean(swe_in_1d[ii], axis=1).reshape(manager_out.lons2D.shape)


    pass

def plot_and_compare_2fields(field1, name1, field2, name2, upper_limit = 1000,
                             basemap = None, manager1 = None, manager2 = None, clevs = None):

    from matplotlib.gridspec import GridSpec
    figure = plt.figure()
    assert isinstance(figure, Figure)
    assert isinstance(manager1, Crcm5ModelDataManager)
    assert isinstance(manager2, Crcm5ModelDataManager)

    if clevs is None:
        clevs = [0, 0.1, 0.5, 1, 10, 100, 300, 500, 800, 1000]
    cmap = cm.get_cmap(name = "jet", lut = len(clevs) - 1)
    norm = BoundaryNorm(clevs, cmap.N)


    vmin = np.min([ np.min(field1), np.min(field2) ])
    vmax = np.max([ np.max(field1), np.max(field2) ])
    vmax = min(vmax, upper_limit)





    gs = GridSpec(1,3, width_ratios=[1,1,0.1])
    ax1 = figure.add_subplot(gs[0, 0])
    assert isinstance(ax1, Axes)
    ax1.set_title(name1)
    x1, y1 = basemap(manager1.lons2D, manager1.lats2D)
    basemap.pcolormesh(x1, y1, field1, cmap=cmap, norm=norm, ax = ax1)

    ax2 = figure.add_subplot(gs[0, 1])
    ax2.set_title(name2)
    x2, y2 = basemap(manager2.lons2D, manager2.lats2D)
    img2 = basemap.pcolormesh(x2, y2, field2, cmap=cmap, norm=norm, ax = ax2)

    ax3 = figure.add_subplot(gs[0, 2])
    plt.colorbar(img2, cax=ax3)


    for ax in figure.get_axes():
        if ax != ax3:
            basemap.drawcoastlines(ax = ax)


    pass


def main():

    swe_obs_manager = SweDataManager(var_name="SWE")

    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3"
    coord_file = os.path.join(data_path, "pm1985050100_00000000p")
    managerLowRes = Crcm5ModelDataManager(samples_folder_path=data_path,
                file_name_prefix="pm", all_files_in_samples_folder=True, need_cell_manager=True
    )

    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_without_lakes_v3"
    coord_file = os.path.join(data_path, "pm1985050100_00000000p")
    managerHighRes = Crcm5ModelDataManager(samples_folder_path=data_path,
                file_name_prefix="pm", all_files_in_samples_folder=True, need_cell_manager=True
    )


    start_year = 1987
    end_year = 1987
    months = [1,2,12]
    rot_lat_lon = RotatedLatLon(lon1=-68, lat1=52, lon2=16.65, lat2=0.0)

    basemap = Basemap(
        projection="omerc",
        llcrnrlon=managerHighRes.lons2D[0,0],
        llcrnrlat=managerHighRes.lats2D[0, 0],
        urcrnrlon=managerHighRes.lons2D[-1,-1],
        urcrnrlat=managerHighRes.lats2D[-1,-1],
        lat_1=rot_lat_lon.lat1,
        lat_2=rot_lat_lon.lat2,
        lon_1=rot_lat_lon.lon1,
        lon_2=rot_lat_lon.lon2,
        no_rot=True
    )

    swe_obs = swe_obs_manager.get_mean(start_year, end_year, months=months)



    obs_ihr = swe_obs_manager.interpolate_data_to(swe_obs, managerHighRes.lons2D,
                                                           managerHighRes.lats2D, nneighbours=1)

    obs_ilr = swe_obs_manager.interpolate_data_to(swe_obs,managerLowRes.lons2D,
                                                          managerLowRes.lats2D, nneighbours=1)

    lowResSwe = managerLowRes.get_mean_field(start_year, end_year, months=months, var_name="I5")



    lowResErr = (lowResSwe - obs_ilr)
    lowResErr[obs_ilr > 0] /= obs_ilr[obs_ilr > 0]
    lowResErr = np.ma.masked_where(obs_ilr <= 0, lowResErr)

    highResSwe = managerHighRes.get_mean_field(start_year, end_year, months= months, var_name="I5")
    highResErr = (highResSwe - obs_ihr)
    highResErr[obs_ihr > 0 ] /= obs_ihr[obs_ihr > 0]
    highResErr = np.ma.masked_where(obs_ihr <= 0, highResErr)


    upscaledHighResSwe = upscale(managerHighRes, managerLowRes, highResSwe)
    upscaledHighResErr = upscaledHighResSwe - obs_ilr
    good_points = obs_ilr > 0
    upscaledHighResErr[good_points] /= obs_ilr[good_points]
    upscaledHighResErr = np.ma.masked_where(~good_points, upscaledHighResErr)



    plot_and_compare_2fields(lowResSwe, "low res", upscaledHighResSwe, "high res (upscaled)", basemap=basemap,
        manager1 = managerLowRes, manager2 = managerLowRes)

    plot_and_compare_2fields(lowResErr, "low res err", upscaledHighResErr, "high res (upscaled) err", basemap=basemap,
        manager1 = managerLowRes, manager2 = managerLowRes, clevs=np.arange(-1, 1.2, 0.2))


    plot_and_compare_2fields(lowResSwe, "low res", highResSwe, "high res", basemap=basemap,
        manager1 = managerLowRes, manager2 = managerHighRes)

    plot_and_compare_2fields(lowResErr, "low res err", highResErr, "high res err", basemap=basemap,
        manager1 = managerLowRes, manager2 = managerHighRes, clevs = np.arange(-1, 1.2, 0.2))

    plt.show()


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print("Hello world")
  
