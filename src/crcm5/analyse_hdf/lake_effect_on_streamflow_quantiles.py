import os
from datetime import datetime
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties
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
    start_year = 1980
    end_year = 2010

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    ids_with_lakes_upstream = [
        "104001", "093806", "093801", "081002", "081007", "080718"
    ]

    selected_station_ids = ["092715", "074903", "080104", "081007", "061905",
                            "093806", "090613", "081002", "093801", "080718", "104001"]


    selected_station_ids = ids_with_lakes_upstream

    #Get the list of stations to do the comparison with
    stations = cehq_station.read_station_data(
        start_date=start_date,
        end_date=end_date,
        selected_ids=selected_station_ids
    )


    #add hydat stations
    # province = "QC"
    # min_drainage_area_km2 = 10000.0
    # stations_hd = cehq_station.load_from_hydat_db(start_date=start_date, end_date=end_date,
    #                                               province=province, min_drainage_area_km2=min_drainage_area_km2)
    # if not len(stations_hd):
    #     print "No hydat stations satisying the conditions: period {0}-{1}, province {2}".format(
    #         str(start_date), str(end_date), province
    #     )
    # stations.extend(stations_hd)


    path1 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r.hdf5"
    label1 = "CRCM5-HCD-R"
    color1 = "b"

    path2 = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"
    label2 = "CRCM5-HCD-RL"
    color2 = "r"

    fldirs = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_FLOW_DIRECTIONS_NAME)
    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(path1)

    lake_fractions = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_LAKE_FRACTION_NAME)
    #cell_areas = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_CELL_AREA_NAME)
    acc_area = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_ACCUMULATION_AREA_NAME)

    cell_manager = CellManager(fldirs, lons2d=lons2d, lats2d=lats2d, accumulation_area_km2=acc_area)

    station_to_mp = cell_manager.get_model_points_for_stations(station_list=stations,
                                                               lake_fraction=lake_fractions,
                                                               drainaige_area_reldiff_limit=0.3)

    fig, axes = plt.subplots(2, 1)

    q90_obs_list = []
    q90_mod1_list = []
    q90_mod2_list = []

    q10_obs_list = []
    q10_mod1_list = []
    q10_mod2_list = []

    for the_station, the_mp in station_to_mp.iteritems():
        assert isinstance(the_station, Station)
        compl_years = the_station.get_list_of_complete_years()
        if len(compl_years) < 3:
            continue

        t, stfl1 = analysis.get_daily_climatology_for_a_point(path=path1, years_of_interest=compl_years,
                                                              i_index=the_mp.ix, j_index=the_mp.jy, var_name="STFA")

        _, stfl2 = analysis.get_daily_climatology_for_a_point(path=path2, years_of_interest=compl_years,
                                                              i_index=the_mp.ix, j_index=the_mp.jy, var_name="STFA")

        _, stfl_obs = the_station.get_daily_climatology_for_complete_years(stamp_dates=t, years=compl_years)

        #Q90
        q90_obs = np.percentile(stfl_obs, 90)
        q90_mod1 = np.percentile(stfl1, 90)
        q90_mod2 = np.percentile(stfl2, 90)

        #Q10
        q10_obs = np.percentile(stfl_obs, 10)
        q10_mod1 = np.percentile(stfl1, 10)
        q10_mod2 = np.percentile(stfl2, 10)

        #save quantiles to lists for correlation calculation
        q90_obs_list.append(q90_obs)
        q90_mod1_list.append(q90_mod1)
        q90_mod2_list.append(q90_mod2)

        q10_mod1_list.append(q10_mod1)
        q10_mod2_list.append(q10_mod2)
        q10_obs_list.append(q10_obs)


        #axes[0].annotate(the_station.id, (q90_obs, np.percentile(stfl1, 90)))
        #axes[1].annotate(the_station.id, (q10_obs, np.percentile(stfl1, 10)))

    #Plot scatter plot of Q90
    the_ax = axes[0]
    #the_ax.annotate(the_station.id, (q90_obs, np.percentile(stfl1, 90)))
    the_ax.scatter(q90_obs_list, q90_mod1_list, label=label1, c=color1)
    the_ax.scatter(q90_obs_list, q90_mod2_list, label=label2, c=color2)

    #plot scatter plot of Q10
    the_ax = axes[1]
    #the_ax.annotate(the_station.id, (q10_obs, np.percentile(stfl1, 10)))
    h1 = the_ax.scatter(q10_obs_list, q10_mod1_list, label=label1, c=color1)
    h2 = the_ax.scatter(q10_obs_list, q10_mod2_list, label=label2, c=color2)



    ##Add correlation coefficients to the axes
    fp = FontProperties(size=20, weight="bold")
    axes[0].annotate(r"$R^2 = {0:.2f}$".format(np.corrcoef(q90_mod1_list, q90_obs_list)[0, 1] ** 2),
                     (0.1, 0.85), color = color1, xycoords = "axes fraction", font_properties = fp)
    axes[0].annotate(r"$R^2 = {0:.2f}$".format(np.corrcoef(q90_mod2_list, q90_obs_list)[0, 1] ** 2),
                     (0.1, 0.75), color = color2, xycoords = "axes fraction", font_properties = fp)

    axes[1].annotate(r"$R^2 = {0:.2f}$".format(np.corrcoef(q10_mod1_list, q10_obs_list)[0, 1] ** 2),
                     (0.1, 0.85), color = color1, xycoords = "axes fraction", font_properties = fp)
    axes[1].annotate(r"$R^2 = {0:.2f}$".format(np.corrcoef(q10_mod2_list, q10_obs_list)[0, 1] ** 2),
                     (0.1, 0.75), color = color2, xycoords = "axes fraction", font_properties = fp)

    for the_ax in axes:
        plot_one_to_one_line(the_ax)
        the_ax.set_xlabel(r"Obs. ${\rm m^3/s}$")
        the_ax.set_ylabel(r"Mod. ${\rm m^3/s}$")

    fig.legend([h1, h2], [label1, label2], loc="upper center")
    figpath = os.path.join(images_folder, "percentiles_comparison.jpeg")
    fig.savefig(figpath, dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    main()