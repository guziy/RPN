from netCDF4 import Dataset
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os
from crcm5.model_data import Crcm5ModelDataManager
from rpn import level_kinds

__author__ = 'huziy'

import numpy as np

##Plot list of fields for one simulation

start_year = 1979
end_year = 1988

field_names = ["TT", "PR", "AU", "AV", "STFL","STFA", "TRAF", "TDRA", "AH"]
file_name_prefixes = ["dm", "pm", "pm", "pm", "pm", "pm", "pm", "pm", "pm"]
sim_name = "crcm5-hcd-rl-intfl"
rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup2/Samples_all_in_one".format(sim_name)
nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

export_to_nc = True


def export_monthly_means_to_ncdb(data_manager, varname, level = -1, level_kind = -1):
    assert isinstance(data_manager, Crcm5ModelDataManager)

    data_manager.export_monthly_mean_fields( sim_name = sim_name, in_file_prefix = data_manager.file_name_prefix,
                                       start_year = start_year, end_year = end_year,
                                       varname = varname, nc_db_folder = nc_db_folder,
                                       level = level, level_kind = level_kind)
    pass

def plot_fields():
    pass


def main():
    dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm", all_files_in_samples_folder=True)
    pmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="pm", all_files_in_samples_folder=True)


    #export monthly means to netcdf files if necessary
    if export_to_nc:
        for varname, prefix in zip( field_names, file_name_prefixes ):
            manager = None
            if prefix == "dm":
                manager = dmManager
            elif prefix == "pm":
                manager = pmManager

            level = -1
            level_kind = -1

            if varname == "TT":
                level = 1
                level_kind = level_kinds.HYBRID


            if varname in ["TRAF", "TDRA"]:
                level = 1
                level_kind = level_kinds.ARBITRARY


            if varname == "STFA": continue
            export_monthly_means_to_ncdb(manager, varname, level= level, level_kind= level_kind)
    #plot results
    assert isinstance(pmManager, Crcm5ModelDataManager)
    lons, lats = pmManager.lons2D, pmManager.lats2D

    basemap = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(
        lons2d=lons, lats2d = lats
    )
    x, y = basemap(lons, lats)


    nc_data_folder = os.path.join(nc_db_folder, sim_name)

    import matplotlib.pyplot as plt
    all_axes = []
    ax_to_levels = {}
    imgs = []

    gs = gridspec.GridSpec(2,3, height_ratios=[1,1], width_ratios=[1,1,1])
    fig = plt.figure()
    #fig.suptitle("({0} - {1})".format(start_year, end_year))
    #plot Temp
    varname = "TT"
    levels = [-30, -25, -10, -5, -2, 0, 2, 5,10, 15, 20, 25]
    cmap = cm.get_cmap("jet", len(levels) - 1)
    bn = BoundaryNorm(levels, cmap.N)
    ds = Dataset(os.path.join(nc_data_folder, "{0}.nc4".format(varname)))
    years = ds.variables["year"][:]
    sel = np.where((start_year <= years) & (years <= end_year))[0]
    tt = ds.variables[varname][sel,:,:,:].mean(axis = 0).mean(axis = 0)
    ds.close()
    ax = fig.add_subplot(gs[0,0])
    ax.set_title("Temperature (${\\rm ^\circ C}$)")
    img = basemap.contourf(x, y, tt, levels = levels, cmap = cmap, norm = bn)
    all_axes.append(ax)
    imgs.append(img)
    ax_to_levels[ax] = levels


    #plot precip
    varname = "PR"
    levels = np.arange(0, 6.5, 0.5)
    cmap = cm.get_cmap("jet_r", len(levels) - 1)
    bn = BoundaryNorm(levels, cmap.N)
    ds = Dataset(os.path.join(nc_data_folder, "{0}.nc4".format(varname)))
    years = ds.variables["year"][:]
    sel = np.where((start_year <= years) & (years <= end_year))[0]
    pr = ds.variables[varname][sel,:,:,:].mean(axis = 0).mean(axis = 0)
    convert_factor = 1000.0 * 24 * 60 * 60  #m/s to mm/day
    pr *= convert_factor
    ds.close()
    ax = fig.add_subplot(gs[0,1])
    ax.set_title("Precip (mm/day)")
    img = basemap.contourf(x, y, pr, levels = levels, cmap = cmap, norm = bn)
    all_axes.append(ax)
    imgs.append(img)
    ax_to_levels[ax] = levels


    #plot AH
    varname = "AH"
    ds = Dataset(os.path.join(nc_data_folder, "{0}.nc4".format(varname)))
    years = ds.variables["year"][:]
    sel = np.where((start_year <= years) & (years <= end_year))[0]
    ah = ds.variables[varname][sel,:,:,:].mean(axis = 0).mean(axis = 0)
    ds.close()

    levels = np.arange(60, 160, 10)# np.linspace(au.min(), au.max(), 10)
    cmap = cm.get_cmap("jet", len(levels) - 1)
    bn = BoundaryNorm(levels, cmap.N)
    ax = fig.add_subplot(gs[1,0])
    ax.set_title("Sensible heat flux (${\\rm W/m^2}$)")
    img = basemap.contourf(x, y, ah,  cmap = cmap)
    all_axes.append(ax)
    imgs.append(img)
    ax_to_levels[ax] = levels

    #plot AV
    varname = "AV"
    ds = Dataset(os.path.join(nc_data_folder, "{0}.nc4".format(varname)))
    years = ds.variables["year"][:]
    sel = np.where((start_year <= years) & (years <= end_year))[0]
    av = ds.variables[varname][sel,:,:,:].mean(axis = 0).mean(axis = 0)
    ds.close()
    levels = np.array( [0,0.1,0.2,0.4, 0.6, 0.8, 1,1.2, 1.4, 1.6, 1.8, 2] )
    levels *= 1e-2
    cmap = cm.get_cmap("jet", len(levels) - 1)
    bn = BoundaryNorm(levels, cmap.N)
    ax = fig.add_subplot(gs[1,1])
    ax.set_title("Latent heat flux (${\\rm W/m^2}$)")
    img = basemap.contourf(x, y, av, levels = levels, cmap = cmap, norm = bn)
    all_axes.append(ax)
    imgs.append(img)
    ax_to_levels[ax] = levels


#plot stfl
    varname = "STFL"
    ds = Dataset(os.path.join(nc_data_folder, "{0}.nc4".format(varname)))
    years = ds.variables["year"][:]
    sel = np.where((start_year <= years) & (years <= end_year))[0]

    stfl = ds.variables[varname][sel,:,:,:].mean(axis = 0).mean(axis = 0)
    ds.close()
    levels = [0,50,100,200,300,500,750,1000, 1500,2000,5000,10000,15000]
    stfl = np.ma.masked_where(stfl < 0.01, stfl)
    cmap = cm.get_cmap("jet", len(levels) - 1)
    bn = BoundaryNorm(levels, cmap.N)
    ax = fig.add_subplot(gs[0,2])
    ax.set_title("Streamflow (${\\rm m^3/s}$)")
    img = basemap.contourf(x, y, stfl, levels = levels, cmap = cmap, norm = bn)
    all_axes.append(ax)
    imgs.append(img)
    ax_to_levels[ax] = levels


    sf  = ScalarFormatter(useMathText=True)
    sf.set_powerlimits([-3,4])



    #draw coast lines
    for the_ax, the_img in zip(all_axes, imgs):
        basemap.drawcoastlines(ax = the_ax)
        divider = make_axes_locatable(the_ax)
        cax = divider.append_axes("right", "10%", pad="3%")

        cb = plt.colorbar(the_img, cax = cax)
        assert isinstance(cax, Axes)
        title = cax.get_title()

    fig.tight_layout()
    fig.savefig("{0}-mean-annual-fields.pdf".format(sim_name))

def doAll():
    global sim_name, rpn_folder
    sims = ["crcm5-hcd-rl-intfl"]
    for sim in sims:
        sim_name = sim
        rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup".format(sim_name)
        main()

if __name__ == "__main__":
    import application_properties
    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=40, height_cm=25)
    application_properties.set_current_directory()

    #doAll()
    main()
    print "Hello world"
  