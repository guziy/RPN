import os
from datetime import datetime
from matplotlib.axes import Axes
from crcm5 import infovar
from data import cehq_station
from data.cehq_station import Station
from data.cell_manager import CellManager
import do_analysis_using_pytables as analysis

import matplotlib.pyplot as plt
import numpy as np

__author__ = 'huziy'

images_folder = "/home/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/images_for_lake-river_paper"

import common_plot_params as cpp


def plot_one_to_one_line(the_ax):
    assert isinstance(the_ax, Axes)
    x1, x2 = the_ax.get_xlim()
    y1, y2 = the_ax.get_ylim()
    lims = [x1, x2, y1, y2]
    z = min(lims), max(lims)
    the_ax.plot(z, z, "-.k")


def main():
    start_year = 1979
    end_year = 1985


    selected_station_ids = [
        "061905", "074903", "090613", "092715", "093801", "093806"
    ]

    #Get the list of stations to do the comparison with
    stations = cehq_station.read_station_data(
        start_date=datetime(start_year, 1, 1),
        end_date=datetime(end_year, 12, 31),
        selected_ids=selected_station_ids
    )


    path1 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf"
    label1 = "CRCM5-HCD-R"

    path2 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf"
    label2 = "CRCM5-HCD-RL"


    fldirs = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_FLOW_DIRECTIONS_NAME)
    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(path1)

    lake_fractions = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_LAKE_FRACTION_NAME)
    #cell_areas = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_CELL_AREA_NAME)
    acc_area = analysis.get_array_from_file(path = path1, var_name=infovar.HDF_ACCUMULATION_AREA_NAME)


    cell_manager = CellManager(fldirs, lons2d=lons2d, lats2d=lats2d, accumulation_area_km2=acc_area)

    station_to_mp = cell_manager.get_model_points_for_stations(station_list=stations, lake_fraction=lake_fractions)

    fig, axes = plt.subplots(2, 1)

    for the_station, the_mp in station_to_mp.iteritems():
        assert isinstance(the_station, Station)
        compl_years = the_station.get_list_of_complete_years()
        if len(compl_years) < 3:
            continue

        t, stfl1 = analysis.get_daily_means_for_a_point(path = path1, years_of_interest=compl_years,
                                                        i_index=the_mp.ix, j_index=the_mp.jy)

        _, stfl2 = analysis.get_daily_means_for_a_point(path = path2, years_of_interest=compl_years,
                                                        i_index=the_mp.ix, j_index=the_mp.jy)

        _, stfl_obs = the_station.get_daily_climatology_for_complete_years(stamp_dates=t, years=compl_years)

        #Q90
        q90_obs = np.percentile(stfl_obs, 90)
        the_ax = axes[0]
        #the_ax.annotate(the_station.id, (q90_obs, np.percentile(stfl1, 90)))
        the_ax.scatter(q90_obs, np.percentile(stfl1, 90), label = label1, c = "b")
        the_ax.scatter(q90_obs, np.percentile(stfl2, 90), label = label2, c = "r")

        #Q10
        q10_obs = np.percentile(stfl_obs, 10)
        the_ax = axes[1]
        #the_ax.annotate(the_station.id, (q10_obs, np.percentile(stfl1, 10)))
        h1 = the_ax.scatter(q10_obs, np.percentile(stfl1, 10), label = label1, c = "b")
        h2 = the_ax.scatter(q10_obs, np.percentile(stfl2, 10), label = label2, c = "r")


    for the_ax in axes:
        plot_one_to_one_line(the_ax)
        the_ax.set_xlabel(r"Obs. ${\rm m^3/s}$")
        the_ax.set_ylabel(r"Mod. ${\rm m^3/s}$")

    fig.legend([h1, h2], [label1, label2], loc = "upper center")
    figpath = os.path.join(images_folder, "percentiles_comparison.jpeg")
    fig.savefig(figpath, dpi = cpp.FIG_SAVE_DPI, bbox_inches = "tight")



if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()