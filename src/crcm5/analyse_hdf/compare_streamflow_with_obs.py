from datetime import datetime
import itertools
import brewer2mpl
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.figure import Figure
import os
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.cehq_station import Station
from data.cell_manager import CellManager
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec

__author__ = 'huziy'

import common_plot_params as cpp

import do_analysis_using_pytables as analysis
#TODO: this is to compare streamflow simulation results with station data
#   input: station ids or list of station objects and
#          the list of simulations, to compare with

#Notes: All the model simulations are assumed to be on the same grid

images_folder = "/home/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/images_for_lake-river_paper"


def _plot_station_position(ax, the_station, basemap, cell_manager, model_point):
    assert isinstance(the_station, Station)
    assert isinstance(model_point, ModelPoint)
    assert isinstance(cell_manager, CellManager)

    x, y = basemap(the_station.longitude, the_station.latitude)
    basemap.scatter(x,y, c = "b", s=25, ax = ax, linewidths = 0)



    #plot the arrows for upstream cells
    ups_mask = cell_manager.get_mask_of_cells_connected_with_by_indices(model_point.ix, model_point.jy)




    basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)



def main(hdf_folder = "/home/huziy/skynet3_rech1/hdf_store"):

    import application_properties
    application_properties.set_current_directory()


    start_date = datetime(1979, 1, 1)
    end_date = datetime(1988, 12, 31)

    # Station ids to get from the CEHQ database
    selected_ids = ["092715", "080101", "074903", "050304", "080104", "081007", "061905",
                    "041903", "040830", "093806", "090613", "081002", "093801", "080718"]


    sim_name_to_file_name = {
        "CRCM5-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf",
        "CRCM5-HCD-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf",
        "CRCM5-HCD-RL": "quebec_0.1_crcm5-hcd-rl_spinup.hdf",
        "CRCM5-HCD-RL-KD5": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl-kd5_spinup.hdf",
        "CRCM5-HCD-RL-INTFL": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup4.hdf"
    }

    #Get the list of stations to do the comparison with
    stations = cehq_station.read_station_data(
        start_date = start_date, end_date = end_date, selected_ids = selected_ids
    )

    stations_hd = cehq_station.load_from_hydat_db(start_date = start_date, end_date = end_date, province="QC")
    stations.extend(stations_hd)

    stations_hd = cehq_station.load_from_hydat_db(start_date = start_date, end_date = end_date, province="ON")
    stations.extend(stations_hd)




    path0 = os.path.join(hdf_folder, sim_name_to_file_name.items()[0][1])
    flow_directions = analysis.get_array_from_file(path=path0, var_name="flow_direction")
    lake_fraction = analysis.get_array_from_file(path=path0, var_name="lake_fraction")
    accumulation_area_km2 = analysis.get_array_from_file(path=path0, var_name="accumulation_area_km2")


    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=path0)

    cell_manager = CellManager(flow_directions,accumulation_area_km2 = accumulation_area_km2,
                               lons2d = lons2d, lats2d = lats2d)

    #Get the list of the corresponding model points
    station_to_modelpoint = cell_manager.get_model_points_for_stations(station_list=stations,
                                                                       lake_fraction=lake_fraction)

    # brewer2mpl.get_map args: set name  set type  number of colors
    bmap = brewer2mpl.get_map("Set1", "qualitative", 9)
    # Change the default colors
    mpl.rcParams["axes.color_cycle"] = bmap.mpl_colors

    for the_station, the_model_point in station_to_modelpoint.iteritems():
        assert isinstance(the_station, Station)
        year_list = the_station.get_list_of_complete_years()
        year_list = list(itertools.ifilter(lambda y: start_date.year <= y <= end_date.year, year_list))


        if len(year_list) <= 3: continue

        fig = plt.figure()

        gs = gridspec.GridSpec(2,3)

        #plot streamflows
        ax = fig.add_subplot(gs[0,:2])

        dates = None
        #get model data for the list of years
        for label, fName in sim_name_to_file_name.iteritems():
            fPath = os.path.join(hdf_folder, fName)

            dates, values_model = analysis.get_daily_means_for_a_point(path=fPath,
                            var_name="STFL",
                            years_of_interest = year_list,
                            i_index = the_model_point.ix,
                            j_index = the_model_point.jy)

            ax.plot(dates, values_model, label = label, lw = 2)

        dates, values_obs = the_station.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=dates,
                    years=year_list)
        ax.plot(dates, values_obs, label = "Obs.", lw = 2)
        ax.set_ylabel("Streamflow: ${\\rm m^3/s}$")
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)
        ax.text(0.1, 0.9, the_station.id, transform = ax.transAxes, bbox = dict(facecolor = "white"))
        ax.legend(loc = (0.0, 1.05), borderaxespad = 0, ncol = 3)
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,2)))
        ax.grid()

        #plot station position
        ax = fig.add_subplot(gs[0, 2:])
        _plot_station_position(ax, the_station, basemap, cell_manager, the_model_point)


        imName = "comp_point_with_obs_{0}_{1}.jpeg".format(the_station.id, the_station.source)
        imFolderPath = os.path.join(images_folder, the_station.source)
        #create a folder for a given source of observed streamflow if it does not exist yet
        if not os.path.isdir(imFolderPath):
            os.mkdir(imFolderPath)

        imPath = os.path.join(imFolderPath, imName)

        fig.savefig(imPath, dpi = cpp.FIG_SAVE_DPI, bbox_inches = "tight")
        plt.close(fig)
