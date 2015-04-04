from datetime import datetime
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.basemap import maskoceans, shiftgrid
from crcm5.model_data import Crcm5ModelDataManager
import matplotlib.pyplot as plt
from narccap.manager import NarccapDataManager
from util import plot_utils

__author__ = 'huziy'

import numpy as np




AGGREGATED_LEVEL = 5
SUBSROF_VARNAME = "TDRA"
SROF_VARNAME = "TRAF"



def plot_ratio_for_the_run(run_name, run_folder, ax, months = list(range(1,13)),
                           subsrof_varname = "TDRA", srof_varname = "TRAF",
                           start_year = -np.Inf, end_year = np.Inf
                           ):
    """
    :type ax: matplotlib.axes.Axes
    """
    dm = Crcm5ModelDataManager(samples_folder_path=run_folder, all_files_in_samples_folder=True)

    #calculate means
    ssrof = dm.get_mean_field(start_year, end_year, months= months, var_name=subsrof_varname, level=AGGREGATED_LEVEL)
    srof = dm.get_mean_field(start_year, end_year, months=months, var_name=srof_varname, level = AGGREGATED_LEVEL)


    trof = srof + ssrof
    srof = np.ma.masked_where((srof < 0) | (trof == 0), srof)
    trof = srof + ssrof

    ratio = ssrof / trof

    #ratio = np.ma.masked_where((srof < 0) | (ssrof < 0), ratio)


    ratio = maskoceans(dm.lons2D , dm.lats2D, ratio, grid=1.25, resolution="i", inlands=False)
    print(ratio.min(), ratio.max())

    b = dm.get_omerc_basemap()
    x, y = b(dm.lons2D, dm.lats2D)
    #levels = [0,0.01,0.05,0.1,0.2,0.4,0.5, 0.6, 0.7, 0.8, 0.9, 1]
    levels = np.arange(0,1.1,0.1)
    cmap = cm.get_cmap("jet", len(levels) - 1 )
    norm = BoundaryNorm(levels, len(levels) - 1)
    img = b.contourf(x, y, ratio, levels = levels, norm = norm, cmap = cmap)
    b.drawcoastlines()

    #ax.figure.colorbar(img)
    ax.set_title(run_name)

    return img, levels









def plot_ratios_using_narccap_data(start_year = None, end_year = None):
    """

    """
    #crcm5 simulation to compare with narccap
    crcm5_sim_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_lowres_002"
    dm = Crcm5ModelDataManager(samples_folder_path=crcm5_sim_path, all_files_in_samples_folder=True)

    model_lons, model_lats = dm.lons2D, dm.lats2D
    basemap = dm.get_rotpole_basemap()
    x, y = basemap(model_lons, model_lats)

    #narccap simulations gcm-rcm names
    narccap_sims = ["ccsm-crcm",  "cgcm3-crcm", "ccsm-wrfg",
                    "cgcm3-rcm3", "gfdl-rcm3", "cgcm3-wrfg",
                    "gfdl-hrm3",  "hadcm3-hrm3",  "gfdl-ecp2"]



    ncManager = NarccapDataManager()

    #plot monthly means

    levels = np.arange(0,1.1,0.1)


    cmap = cm.get_cmap("jet", len(levels) - 1 )
    norm = BoundaryNorm(levels, len(levels) - 1)

    plot_utils.apply_plot_params(width_pt=None, width_cm=22, height_cm=30)
    ncols = 3
    nrows = 4
    img = None
    for m in range(1,13):
        fig = plt.figure()
        gs = gridspec.GridSpec(nrows,ncols, height_ratios=[1,1,1,0.5])
        i = 0 #just in case we don't go inside the loop
        for i, the_sim in enumerate(narccap_sims):
            print("processing {0} ...".format(the_sim))
            gcm, rcm = the_sim.split("-")
            ax = fig.add_subplot(gs[i//ncols, i%ncols])

            tot_rof = ncManager.get_climatologic_field(varname="mrro", gcm=gcm, rcm=rcm,
                start_year=start_year, end_year= end_year, months=[m,]
            )

            surf_rof = ncManager.get_climatologic_field(varname="mrros", gcm=gcm, rcm=rcm,
                            start_year=start_year, end_year= end_year, months=[m,]
            )

            good_points = (tot_rof > 0) & (surf_rof >= 0) & (tot_rof >= surf_rof)
            data = (tot_rof - surf_rof)/tot_rof
            data = np.ma.masked_where(~good_points, data)

            #interpolate to the model grid
            data = ncManager.inerpolate_to(model_lons, model_lats, data)
            data = maskoceans(model_lons, model_lats, data)
            img = basemap.contourf(x, y, data, levels = levels, cmap = cmap, ax = ax)
            basemap.drawcoastlines(ax = ax)
            ax.set_title(the_sim.upper())

        #plot the panel with model data
#        i += 1
#        ax = fig.add_subplot(gs[i//ncols, i%ncols])
#        ssrof = dm.get_mean_field(start_year, end_year, months= [m,],
#            var_name=SUBSROF_VARNAME, level=AGGREGATED_LEVEL)
#        srof = dm.get_mean_field(start_year, end_year, months= [m,],
#            var_name=SROF_VARNAME, level = AGGREGATED_LEVEL)
#
#
#        trof = srof + ssrof
#        srof = np.ma.masked_where((srof < 0) | (trof == 0), srof)
#        trof = srof + ssrof
#
#        ratio = ssrof / trof
#        img = basemap.contourf(x, y, ratio, levels = levels, cmap = cmap, ax = ax)
#        basemap.drawcoastlines(ax = ax)
#        ax.set_title("era40-crcm5")


        #plot color bar
        i += 1
        ax = fig.add_subplot(gs[i // ncols, (i % ncols):])
        assert isinstance(ax, Axes)
        ax.set_aspect(1.0/20.0)

        fig.colorbar(img, ticks = levels[::2] ,cax = ax, orientation = "horizontal")
        ax.set_title(datetime(2000,m,1).strftime("%B"))
        fig.tight_layout()
        fig.savefig("{0}_month_rof_ratio.png".format(m))


    pass




def plot_ratio_for_all_seasons():
    fig = plt.figure()
    run_name = "CRCM5-HCD-RL"
    run_folder = ""

    seasons = [
            [12,1,2],
            list(range(3,6)),
            list(range(6,9)),
            list(range(9,12))
    ]


    gs = gridspec.GridSpec(2,len(seasons), height_ratios=[1, 0.03])
    img = None
    levels = None
    i = 0

    for months in seasons:
        ax = fig.add_subplot(gs[0,i])
        img, levels = plot_ratio_for_the_run(run_name, run_folder, ax, start_year=1985, months=months)
        i += 1
    ax = fig.add_subplot(gs[1,:])
    assert isinstance(ax, Axes)
    #ax.set_aspect(1.0/20.0)
    fig.colorbar(img, cax = ax, ticks = levels, orientation = "horizontal")




    pass

def main():

    fig = plt.figure()
    run_names = ["low res. 002 (3 soil layers)", "low res. 001 (10 soil layers)"]
    run_folders = ["/home/huziy/skynet3_exec1/from_guillimin/quebec_lowres_002",
                   "/home/huziy/skynet3_exec1/from_guillimin/quebec_lowres_001"
                   ]


    gs = gridspec.GridSpec(2,len(run_names), height_ratios=[1, 0.03])
    img = None
    levels = None
    i = 0
    for run_name, run_folder in zip(run_names, run_folders):
        ax = fig.add_subplot(gs[0,i])
        img, levels = plot_ratio_for_the_run(run_name, run_folder, ax, start_year=1986, months=[5,])
        i += 1
    ax = fig.add_subplot(gs[1,:])
    assert isinstance(ax, Axes)
    #ax.set_aspect(1.0/20.0)
    fig.colorbar(img, cax = ax, ticks = levels, orientation = "horizontal")
    fig.savefig("ssroff_to_totalrof_ratio_may.jpeg")

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    #main()
    plot_ratios_using_narccap_data(start_year=1985, end_year=1990)
    print("Hello world")
  
