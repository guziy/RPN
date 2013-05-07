from netCDF4 import Dataset
import os
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import ScalarFormatter, MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
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



ncdb_path = "/home/huziy/skynet3_rech1/crcm_data_ncdb"
start_year = 1979
end_year = 1988

do_export = False

def export_means_to_netcdf(varname = "PR", file_prefix = "pm", level = -1, level_kind = level_kinds.ARBITRARY):
    base_data_path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_gemclim_orig_0.5deg"

    base_data_manager = Crcm5ModelDataManager(samples_folder_path=base_data_path, all_files_in_samples_folder=True)
    base_data_manager.export_monthly_mean_fields(sim_name="quebec_gemclim_orig_0.5deg", in_file_prefix=file_prefix,
            start_year = start_year, end_year=end_year, varname= varname, level=level, level_kind=level_kind
    )


    case_data_path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_gemclim_orig_0.1deg"
    case_data_manager = Crcm5ModelDataManager(samples_folder_path=case_data_path, all_files_in_samples_folder=True)
    case_data_manager.export_monthly_mean_fields(sim_name="quebec_gemclim_orig_0.1deg", in_file_prefix=file_prefix,
            start_year = start_year, end_year=end_year, varname= varname, level=level, level_kind=level_kind
    )


def get_cru_obs_mean_fields(target_lon_2d, target_lat_2d, nneighbours = 1, months = None):
    tmpCruDataManager = CRUDataManager(var_name="tmp", lazy = True)
    preCruDataManager = CRUDataManager(path = "data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc", var_name="pre", lazy = True)


    #get mean annual temperature from CRU dataset
    tmpCru = tmpCruDataManager.get_mean(start_year=start_year, end_year = end_year, months = months)
    #same thing for the precip
    preCru = preCruDataManager.get_mean(start_year=start_year, end_year = end_year, months = months) ##pre is in mm/month

    #convert to mm/day
    preCru *= 12.0 / 365.25

    tmpCruI = tmpCruDataManager.interpolate_data_to(tmpCru, target_lon_2d, target_lat_2d, nneighbours=nneighbours)
    preCruI = preCruDataManager.interpolate_data_to(preCru, target_lon_2d, target_lat_2d, nneighbours=nneighbours)

    return tmpCruI, preCruI



def main():
    if do_export:
        export_means_to_netcdf(varname="PR", file_prefix="pm", level=-1)

    base_data_path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_gemclim_orig_0.5deg"
    base_data_manager = Crcm5ModelDataManager(samples_folder_path=base_data_path, all_files_in_samples_folder=True, file_name_prefix="dm")
    b05 = base_data_manager.get_rotpole_basemap()

    case_data_path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_gemclim_orig_0.1deg"
    case_data_manager = Crcm5ModelDataManager(samples_folder_path=case_data_path, all_files_in_samples_folder=True, file_name_prefix="dm")
    b01 = case_data_manager.get_rotpole_basemap()



    months = [6,7,8]

    lowres_nc_folder = os.path.join(ncdb_path, "quebec_gemclim_orig_0.5deg")
    highres_nc_folder = os.path.join(ncdb_path, "quebec_gemclim_orig_0.1deg")

    #read lowres data and coordinates
    ds = Dataset(os.path.join(lowres_nc_folder, "TT.nc4"))
    tt05 = ds.variables["TT"][:,months,:,:].mean(axis=0).mean(axis = 0)

    ds = Dataset(os.path.join(lowres_nc_folder, "PR.nc4"))
    pr05 = ds.variables["PR"][:,months,:,:].mean(axis=0).mean(axis = 0)

    lon05, lat05 = ds.variables["lon"][:], ds.variables["lat"][:]
    x05, y05 = b05(lon05, lat05)


    #read highres data and coordinates
    ds = Dataset( os.path.join(highres_nc_folder, "TT.nc4"))
    tt01 = ds.variables["TT"][:,months,:,:].mean(axis=0).mean(axis = 0)

    ds = Dataset( os.path.join(highres_nc_folder, "PR.nc4"))
    pr01 = ds.variables["PR"][:,months,:,:].mean(axis=0).mean(axis = 0)



    lon01, lat01 = ds.variables["lon"][:], ds.variables["lat"][:]

    plt.pcolormesh(lon01.transpose())
    plt.colorbar()
    plt.show()

    x01, y01 = b01(lon01, lat01)


    #read cru obs data
    tmpCruI05, preCruI05 = get_cru_obs_mean_fields(lon05, lat05, nneighbours=1, months = months)


    #plotting starts here
    fig = plt.figure()

    plt.suptitle("({0} - {1})".format(start_year, end_year))

    nrows = 2
    ncols = 7
    gs = gridspec.GridSpec(nrows = nrows, ncols = ncols, width_ratios=[1,1,1,0.1,1,1,0.1],
        height_ratios=[1,1], left=0.05)



    #plot temperatures
    levels = [-30,-20,-10,-5, -2, 0, 2, 5, 10,15,20,25]
    cMap = get_cmap("jet", len(levels) - 1 )
    bn = BoundaryNorm(levels, cMap.N)


    #plot lowres temp
    ax = fig.add_subplot(gs[0,0])
    img = b05.contourf(x05, y05, tt05, levels = levels, norm = bn, cmap = cMap)
    b05.drawcoastlines()
    ax.set_title("dx = 0.5")
    ax.set_ylabel("TT")

    #plot highres temp
    ax = fig.add_subplot(gs[0,1])
    img = b01.contourf(x01, y01, tt01, levels = levels, norm = bn, cmap = cMap)
    b01.drawcoastlines()
    ax.set_title("dx = 0.1")

    ax = fig.add_subplot(gs[0,2])
    img = b05.contourf(x05, y05, tmpCruI05, levels = levels, norm = bn, cmap = cMap)
    b05.drawcoastlines(ax = ax)
    ax.set_title("CRU")


    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", "5%", pad="3%")
    cax = fig.add_subplot(gs[0,3])
    cax.set_aspect(20)
    cb = fig.colorbar(img,  cax = cax, ticks = levels)
    cax.set_title(" ${\\rm ^{\circ} C}$")

    #plot tt05-tt01
    levels = np.arange(-3,4, 0.5)
    cMap = cm.get_cmap("RdBu", len(levels) - 1)



    tt01I05 = case_data_manager.interpolate_data_to(tt01, lon05, lat05, nneighbours=25)
    ax = fig.add_subplot(gs[0,4])
    img = b05.contourf(x05, y05, tt05 - tt01I05, cmap = cMap, levels = levels)
    b05.drawcoastlines()
    b05.drawmapboundary(fill_color="0.75")
    ax.set_title("TT(dx = 0.5) - TT(dx = 0.1)")



    #plot tt01I05-Cru
    ax = fig.add_subplot(gs[0,5])
    img = b05.contourf(x05, y05, tt01I05 - tmpCruI05, cmap = cMap, levels = levels)
    b05.drawcoastlines()
    b05.drawmapboundary(fill_color="0.75")
    ax.set_title("TT(dx = 0.1) - CRU")



    cax = fig.add_subplot(gs[0,6])
    cax.set_aspect(20)
    cb = fig.colorbar(img,  cax = cax, ticks = levels)
    cax.set_title("${\\rm ^{\circ} C}$")






    #plot precip
    convert_factor = 1000.0 * 24 * 60 * 60  #m/s to mm/day
    units = "mm/day \n"
    levels = np.arange(0,7,0.5)
    cMap = get_cmap("jet", len(levels) - 1 )
    bn = BoundaryNorm(levels, cMap.N)


    pr01 *= convert_factor
    pr05 *= convert_factor


    #plot lowres temp
    ax = fig.add_subplot(gs[1,0])
    img = b05.contourf(x05, y05, pr05, levels = levels, cmap = cMap)
    b05.drawcoastlines()
    ax.set_ylabel("PR")

    #plot highres temp
    ax = fig.add_subplot(gs[1,1])
    img = b01.contourf(x01, y01, pr01, levels = levels, cmap = cMap)
    b01.drawcoastlines()



    ax = fig.add_subplot(gs[1,2])
    img = b05.contourf(x05, y05, preCruI05, levels = levels, norm = bn, cmap = cMap)
    b05.drawcoastlines(ax = ax)
    ax.set_title("CRU")




    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", "5%", pad="3%")
    cax = fig.add_subplot(gs[1,3])
    cax.set_aspect(20)
    cb = fig.colorbar(img,  cax = cax, ticks = levels)
    cax.set_title(units)


    #plot pr05-pr01
    levels = np.arange(-2,3, 0.5)
    cMap = cm.get_cmap("RdBu", len(levels) - 1)
    cMap.set_over(color=cMap(levels[-1]), alpha=1.0)
    cMap.set_under(color=cMap(levels[0]), alpha=1.0)
    cMap.set_bad(color="0.75",alpha=1.0)



    pr01I05 = case_data_manager.interpolate_data_to(pr01, lon05, lat05, nneighbours=25)
    ax = fig.add_subplot(gs[1,4])
    img = b05.contourf(x05, y05, pr05 - pr01I05, cmap = cMap, levels = levels)
    b05.drawcoastlines()
    b05.drawmapboundary(fill_color="0.75")
    ax.set_title("PR(dx = 0.5) - PR(dx = 0.1)")


    #plot tt01I05-Cru
    ax = fig.add_subplot(gs[1,5])
    img = b05.contourf(x05, y05, pr01I05 - preCruI05, cmap = cMap, levels = levels)
    b05.drawcoastlines()
    b05.drawmapboundary(fill_color="0.75")
    ax.set_title("PR(dx = 0.1) - CRU")



    cax = fig.add_subplot(gs[1,6])
    cax.set_aspect(20)
    cb = fig.colorbar(img,  cax = cax, ticks = levels, extend = "both")
    cax.set_title("mm/day \n")



    #fig.tight_layout()
    fig.savefig("high_and_low_res_comparison.pdf")







if __name__ == "__main__":
    plot_utils.apply_plot_params(width_pt=None, width_cm=50, height_cm=20, font_size=12)
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  