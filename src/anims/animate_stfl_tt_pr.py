from datetime import datetime
from netCDF4 import Dataset
import shutil
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os
from crcm5.model_data import Crcm5ModelDataManager


import matplotlib.animation as animation

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt

class Animator():
    """
    Animates monthly means of temperature precipitation and streamflow
    reading variables from the monthly mean files assuming the structure F(year, month, x, y)
    """
    def __init__(self, var_names, ncDatasetsDict, basemap, x, y, stfl_mask = None):
        self.ncDatasets = ncDatasetsDict
        self.temperatureName = "TT"
        self.precipName = "PR"
        self.streamflowName = "STFL"

        self.var_names = var_names

        self.temperatureVar = ncDatasetsDict[self.temperatureName].variables[self.temperatureName]
        self.precipVar = ncDatasetsDict[self.precipName].variables[self.precipName]
        self.streamflowVar = ncDatasetsDict[self.streamflowName].variables[self.streamflowName]


        self.years = ncDatasetsDict[self.temperatureName].variables["year"][:]

        self.basemap = basemap
        self.x = x
        self.y = y

        self.current_frame_index = 0
        self.year_index = 0
        self.month_index = 0


        self.redraw_colorbars = True
        self.title = None

        self.images = None
        self.stfl_mask = stfl_mask

        self.seasons = ["Winter", "Spring", "Summer", "Fall"]




    def animate(self, t):
        print("current frame: {0}".format(self.current_frame_index))
        #clear axes
        #for ax in self.axesDict.values():
        #    ax.cla()


        #prepare figure and subplot for animation
        fig = plt.figure()
        self.figure = fig
        gs = gridspec.GridSpec(2,2, width_ratios=[1,1])


        axesDict = {}
        for i, vName in enumerate(self.var_names):
            if vName == "STFL":
                axesDict[vName] = fig.add_subplot(gs[0,0:2])
            elif vName == "TT":
                axesDict[vName] = fig.add_subplot(gs[1,0])
            elif vName == "PR":
                axesDict[vName] = fig.add_subplot(gs[1,1])

        axesDict["TT"].set_title("Temperature (${\\rm ^\circ C}$)")
        axesDict["PR"].set_title("Precip (mm/day)")
        axesDict["STFL"].set_title("{0}-{1:02d} ({2})".format(t.year, t.month, self.seasons[0 if t.month == 12 else (t.month // 3)]))

        assert isinstance(fig, Figure)


        basemap = self.basemap
        all_axes = []
        ax_to_levels = {}
        imgs = []


        #fig.suptitle()
        #fig.suptitle("({0} - {1})".format(start_year, end_year))
        #plot Temp
        levels = [-40, -30, -20, -10, -5, 0,5, 10, 15, 20, 25, 30]
        cmap = cm.get_cmap("jet", len(levels) - 1)
        bn = BoundaryNorm(levels, cmap.N)


        ax = axesDict["TT"]
        assert isinstance(ax, Axes)

        tt = self.temperatureVar[self.year_index,self.month_index,:,:]
        img = basemap.contourf(self.x, self.y, tt, levels = levels, cmap = cmap, norm = bn, ax = ax)
        all_axes.append(ax)
        imgs.append(img)
        ax_to_levels[ax] = levels



        #plot precip
        ax = axesDict["PR"]
        levels = np.arange(0, 15, 1.5)
        cmap = cm.get_cmap("jet", len(levels) - 1)
        bn = BoundaryNorm(levels, cmap.N)

        convert_factor = 1000.0 * 24 * 60 * 60  #m/s to mm/day
        pr = self.precipVar[self.year_index,self.month_index,:,:] * convert_factor
        img = basemap.contourf(self.x, self.y, pr, levels = levels, cmap = cmap, norm = bn, ax = ax)
        all_axes.append(ax)
        imgs.append(img)
        ax_to_levels[ax] = levels




    #plot stfl
        ax = axesDict["STFL"]
        levels = [0,50,100,200,300,500,750,1000, 1500,2000,5000,10000,15000]
        cmap = cm.get_cmap("jet", len(levels) - 1)
        bn = BoundaryNorm(levels, cmap.N)
        stfl = self.streamflowVar[self.year_index,self.month_index,:,:]
        stfl = np.ma.masked_where(self.stfl_mask, stfl)
        img = basemap.contourf(self.x, self.y, stfl, levels = levels, cmap = cmap, norm = bn, ax = ax)
        all_axes.append(ax)
        imgs.append(img)
        ax_to_levels[ax] = levels


        sf  = ScalarFormatter(useMathText=True)
        sf.set_powerlimits([-3,4])



        #draw coast lines
        for the_ax, the_img in zip(all_axes, imgs):
            basemap.drawcoastlines(ax = the_ax)
            divider = make_axes_locatable(the_ax)

            cax = divider.append_axes("right", "5%", pad="3%")
            cb = plt.colorbar(the_img, cax = cax, ticks = ax_to_levels[the_ax])

            if the_ax == axesDict["STFL"]:
                cb.ax.set_ylabel("Streamflow (${\\rm m^3/s}$)")


        #self.redraw_colorbars = False


        self.current_frame_index += 1

        self.year_index = self.current_frame_index // 12
        self.month_index = self.current_frame_index % 12








    def closeDataConnections(self):
        [ds.close() for ds in self.ncDatasets]

    def saveFrame(self, tmp_folder = "", prefix = "tt_pr_stf_anim_"):
        self.figure.tight_layout()
        self.figure.savefig(os.path.join(tmp_folder, prefix + "{0:08d}.png".format(self.current_frame_index)))
        pass



def main():

    rpn_folder = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup"
    nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"

    start_year = 1979
    end_year = 1988

    sim_name = "crcm5-r"

    #dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm", all_files_in_samples_folder=True)
    pmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="pm", all_files_in_samples_folder=True)


    #plot results
    assert isinstance(pmManager, Crcm5ModelDataManager)
    lons, lats = pmManager.lons2D, pmManager.lats2D


    stfl_mask = pmManager.cbf < 0

    basemap = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(
        lons2d=lons, lats2d = lats
    )
    x, y = basemap(lons, lats)
    print(x.shape)


    month_dates = [ datetime(year, month, 15)  for year in range(start_year, end_year + 1) for month in range(1,13) ]

    print(len(month_dates), " month of animation ")

    nc_data_folder = os.path.join(nc_db_folder, sim_name)
    dsDict = {}
    var_names = [ "STFL", "PR", "TT"]
    for i, vName in enumerate(var_names):
        path = os.path.join(nc_data_folder, "{0}.nc4".format(vName))
        dsDict[vName] = Dataset(path)

    aniObj = Animator(var_names, dsDict, basemap, x, y, stfl_mask = stfl_mask)

    #fa = animation.FuncAnimation(fig, aniObj.animate, month_dates, interval=50)
    temp_folder = "for_anim_{0}".format(sim_name)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)


    for d in month_dates:
        aniObj.animate(d)

        aniObj.saveFrame(tmp_folder=temp_folder)

    #fa.save("animate_stfl_pr_tt.mpg", writer= writer)


if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=30, height_cm=30, font_size=26)
    main()
    print("Hello world")
  