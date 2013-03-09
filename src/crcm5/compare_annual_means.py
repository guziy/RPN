from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.basemap import Basemap
from cru.temperature import CRUDataManager
import my_colormaps
from rpn import level_kinds
from util import plot_utils

__author__ = 'huziy'

import numpy as np

from model_data import Crcm5ModelDataManager
import matplotlib.pyplot as plt
#Produces a plot (v_mean1 - v_mean_base)/v_mean_base for the given 2d variable
#with the option to mask the points where the changes are not significant
#initially designed to compare streamflow, with the base streamflow corresponding to
#the case without lakes and without lake runoff

from matplotlib.cm import get_cmap


def main():


    start_year = 1987
    end_year = 1987

    #paths to the model results
    #base_data_path =  "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff"
    base_data_path =  "/home/huziy/skynet3_exec1/from_guillimin/quebec_lowres_001"
    case_data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_260x260_wo_lakes_and_with_lakeroff_nogw"



    tmpCruDataManager = CRUDataManager(var_name="tmp")
    preCruDataManager = CRUDataManager(path = "data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc", var_name="pre")


    #get mean annual temperature from CRU dataset
    tmpCru = tmpCruDataManager.get_mean(start_year=start_year, end_year = end_year)
    #same thing for the precip
    preCru = preCruDataManager.get_mean(start_year=start_year, end_year = end_year) ##pre is in mm/month


    base_data_manager = Crcm5ModelDataManager(samples_folder_path=base_data_path, all_files_in_samples_folder=True)
    case_data_manager = Crcm5ModelDataManager(samples_folder_path=case_data_path, all_files_in_samples_folder=True)

    baseMeans = base_data_manager.get_annual_mean_fields(varname="PR",start_year=start_year, end_year=end_year)
    caseMeans = case_data_manager.get_annual_mean_fields(varname="PR", start_year=start_year, end_year=end_year)


    lons2d = case_data_manager.lons2D
    lats2d = case_data_manager.lats2D



    b = case_data_manager.get_omerc_basemap()
    x, y = b(lons2d, lats2d)






    #plot precipitation mm/year
    all_axes = []
    fig = plt.figure()
    nrows = 2
    ncols = 7
    gs = gridspec.GridSpec(nrows, ncols)

    levels = [0,100, 200, 300, 400,500,600, 700, 800, 900, 1000, 1200, 1500, 1800,2000, 2500]
    cMap = get_cmap("jet", len(levels) - 1 )
    bn = BoundaryNorm(levels, cMap.N)


    ax = fig.add_subplot(gs[0,0])
    assert isinstance(ax, Axes)
    all_axes.append(ax)
    ax.set_ylabel("PR")
    ax.set_title("CRCM5 (0.5 deg.)")
    base_mean_field = baseMeans.mean()* 86400 * 365 * 1000


    basemap1 = base_data_manager.get_omerc_basemap()
    x1, y1 = basemap1(base_data_manager.lons2D, base_data_manager.lats2D)
    b.pcolormesh(x1, y1, base_mean_field, norm=bn, vmin=0, vmax=levels[-1], cmap = cMap, ax = ax)

    ax = fig.add_subplot(gs[0,1])
    all_axes.append(ax)
    ax.set_title("CRCM5 (0.1 deg.)")
    case_mean_field = caseMeans.mean() * 86400 * 365 * 1000
    b.pcolormesh(x, y, case_mean_field, norm=bn, vmin=0, vmax=levels[-1], cmap = cMap, ax = ax)

    ax = fig.add_subplot(gs[0,2])
    all_axes.append(ax)
    ax.set_title("CRU")
    preCruI = preCruDataManager.interpolate_data_to(preCru, lons2d, lats2d, nneighbours=1)
    img = b.pcolormesh(x, y, preCruI * 12 , norm=bn, vmin=0, vmax=levels[-1], cmap = cMap, ax = ax)

    cax = fig.add_subplot(gs[0,3])
    assert isinstance(cax, Axes)
    cax.set_aspect(20.0)
    fig.colorbar(img, ticks = levels, cax = cax)

    #plot deltas (pcp)
    cmap_diff = cm.get_cmap("jet", 10) #my_colormaps.get_red_blue_colormap()

    ax = fig.add_subplot(gs[0,4])
    assert isinstance(ax, Axes)
    all_axes.append(ax)
    ax.set_ylabel("delta(PR)")
    ax.set_title("CRCM5 (0.5 deg.) - CRU")
    preCruI1 = preCruDataManager.interpolate_data_to(preCru, base_data_manager.lons2D, base_data_manager.lats2D, nneighbours=1)
    bias1 = (base_mean_field - preCruI1)
    bias2 =  (case_mean_field - preCruI)

    #calculate and set limits in order to have a common colorbar
    vmin = min(np.min(bias1), np.min(bias2))
    vmax = max(np.max(bias1), np.max(bias2))
    #limit error to 1000mm
    vmax = min(vmax, 1000.0)

    b.pcolormesh(x1, y1, bias1 , cmap= cmap_diff, vmin = vmin, vmax = vmax)


    ax = fig.add_subplot(gs[0,5])
    assert isinstance(ax, Axes)
    all_axes.append(ax)
    ax.set_title("CRCM5 (0.1 deg.) - CRU")
    img = b.pcolormesh(x, y, bias2 , cmap= cmap_diff, vmin = vmin, vmax = vmax)

    cax = fig.add_subplot(gs[0,6])
    cax.set_aspect(20.0)
    fig.colorbar(img, cax = cax, extend = "max")



    #draw temperatures
    base_data_manager = Crcm5ModelDataManager(samples_folder_path=base_data_path, all_files_in_samples_folder=True,
        file_name_prefix="dm")
    case_data_manager = Crcm5ModelDataManager(samples_folder_path=case_data_path, all_files_in_samples_folder=True,
        file_name_prefix="dm"
    )

    baseMeans = base_data_manager.get_annual_mean_fields(varname="TT",level=1,
        level_kind= level_kinds.HYBRID, start_year=start_year, end_year=end_year)
    caseMeans = case_data_manager.get_annual_mean_fields(varname="TT",level=1,
        level_kind= level_kinds.HYBRID, start_year=start_year, end_year=end_year)

    #baseMeans = base_data_manager.get_mean_2d_field_and_all_data(var_name="TT", )

    levels = [-5, -2, 0, 2, 5, 10,15,20,25]
    cMap = get_cmap("jet", len(levels) - 1 )
    bn = BoundaryNorm(levels, cMap.N)


    ax = fig.add_subplot(gs[1,0])
    assert isinstance(ax, Axes)
    all_axes.append(ax)
    ax.set_ylabel("TT")
    ax.set_title("CRCM5 (0.5 deg.)")
    base_mean_field = baseMeans.mean()


    basemap1 = base_data_manager.get_omerc_basemap()
    x1, y1 = basemap1(base_data_manager.lons2D, base_data_manager.lats2D)
    b.pcolormesh(x1, y1, base_mean_field, norm=bn, vmin=levels[0], vmax=levels[-1], cmap = cMap, ax = ax)

    ax = fig.add_subplot(gs[1,1])
    all_axes.append(ax)
    ax.set_title("CRCM5 (0.1 deg.)")
    case_mean_field = caseMeans.mean()
    b.pcolormesh(x, y, case_mean_field, norm=bn, vmin=levels[0], vmax=levels[-1], cmap = cMap, ax = ax)

    ax = fig.add_subplot(gs[1,2])
    all_axes.append(ax)
    ax.set_title("CRU")
    tmpCruI = preCruDataManager.interpolate_data_to(tmpCru, lons2d, lats2d, nneighbours=1)
    img = b.pcolormesh(x, y, tmpCruI, norm=bn, vmin=levels[0], vmax=levels[-1], cmap = cMap, ax = ax)

    cax = fig.add_subplot(gs[1,3])
    assert isinstance(cax, Axes)
    cax.set_aspect(20.0)
    fig.colorbar(img, ticks = levels, cax = cax)




    #plot deltas (temp)
    cmap_diff = cm.get_cmap("jet", 10) #my_colormaps.get_red_blue_colormap()
    ax = fig.add_subplot(gs[1,4])
    assert isinstance(ax, Axes)
    all_axes.append(ax)
    ax.set_ylabel("delta(TT)")
    ax.set_title("CRCM5 (0.5 deg.) - CRU")
    tmpCruI1 = tmpCruDataManager.interpolate_data_to(tmpCru, base_data_manager.lons2D, base_data_manager.lats2D, nneighbours = 1)

    bias1 = base_mean_field - tmpCruI1
    bias2 = case_mean_field - tmpCruI
    vmin = min(bias1.min(), bias2.min())
    vmax = max(bias1.max(), bias2.max())

    b.pcolormesh(x1, y1, bias1 , cmap= cmap_diff, vmin = vmin, vmax = vmax)

    ax = fig.add_subplot(gs[1,5])
    assert isinstance(ax, Axes)
    all_axes.append(ax)
    ax.set_title("CRCM5 (0.1 deg.) - CRU")
    img = b.pcolormesh(x, y, bias2, cmap= cmap_diff, vmin = vmin, vmax = vmax)

    cax = fig.add_subplot(gs[1,6])
    cax.set_aspect(20.0)
    fig.colorbar(img, cax = cax)






    for the_ax in all_axes:
        b.drawcoastlines(ax = the_ax)
    fig.tight_layout()
    fig.savefig("cmp_high_low_res.jpg")


    return








    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    plot_utils.apply_plot_params(width_cm=120.0, height_cm=20.0, font_size=8)
    main()
    print "Hello world"
  