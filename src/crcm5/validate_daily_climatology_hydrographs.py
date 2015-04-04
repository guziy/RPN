from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, DateLocator, MonthLocator
from matplotlib.font_manager import FontProperties
from crcm5.model_data import Crcm5ModelDataManager
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.cehq_station import Station
import os
from datetime import datetime
__author__ = 'huziy'
import matplotlib.pyplot as plt

import numpy as np
def validate_daily_climatology():
    """

    """
    #years are inclusive
    start_year = 1979
    end_year =1988

    sim_name_list = ["crcm5-r", "crcm5-hcd-r", "crcm5-hcd-rl"]
    rpn_folder_path_form = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup"
    nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"



    #select stations
    selected_ids = None
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    print("start reading cehq obs data")
#    stations = cehq_station.read_station_data(selected_ids = selected_ids,
#            start_date=start_date, end_date=end_date
#    )
    stations = []

    print("start reading hydat obs data")
    stations.extend(cehq_station.read_hydat_station_data(folder_path="/home/huziy/skynet3_rech1/HYDAT",
            start_date = start_date, end_date = end_date))

    print("finished reading station data")

    varname = "STFL"
    sim_name_to_manager = {}
    sim_name_to_station_to_model_point = {}
    dmManager = None



    for sim_name in sim_name_list:
        print(sim_name)
        rpn_folder = rpn_folder_path_form.format(sim_name)

        dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm",
            all_files_in_samples_folder=True, need_cell_manager=True)

        sim_name_to_manager[sim_name] = dmManager

        nc_sim_folder = os.path.join(nc_db_folder, sim_name)
        nc_path = os.path.join(nc_sim_folder, "{0}_all.nc".format(varname))

        print("get model points for the stations")

        st_to_mp = dmManager.get_model_points_for_stations(stations, nc_path=nc_path, npoints=1,
            nc_sim_folder=nc_sim_folder
        )

        sim_name_to_station_to_model_point[sim_name] = st_to_mp


    common_lake_fractions = dmManager.lake_fraction

    #for tests
    #test(sim_name_to_station_to_model_point)

    print("finished reading data in memory")


    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages("comp_with_obs_daily_clim.pdf")


    stations_to_plot = [] #only stations that are compared with model are needed on a map
    day_stamps = Station.get_stamp_days(2001)
    for s in stations:
        plt.figure()

        assert isinstance(s, Station)

        years = s.get_list_of_complete_years()

        print(s)
        if len(years) < 6: continue

        dates, obs_data = s.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=day_stamps, years=years)
        obs_ann_mean = np.mean(obs_data)

        plt.plot( dates, obs_data, label = "Obs: ann.mean = {0:.1f}".format( obs_ann_mean ) )

        mp = None
        for sim_name in sim_name_list:
            manager = sim_name_to_manager[sim_name]
            if s not in sim_name_to_station_to_model_point[sim_name]:
                continue

            mp = sim_name_to_station_to_model_point[sim_name][s]
            for mp in sim_name_to_station_to_model_point[sim_name][s]:
                assert isinstance(mp, ModelPoint)
                dates, values = mp.get_daily_climatology_for_complete_years(stamp_dates=day_stamps, varname = "STFL")
                plt.plot(dates, values , label = "{0}: {1:.2f} \n ann.mean = {2:.1f}, dist = {3:.1f} km".format( sim_name,
                    manager.lake_fraction[mp.flow_in_mask == 1].mean(), np.mean(values), mp.distance_to_station / 1000.0) )

                ax = plt.gca()
                assert isinstance(ax, Axes)

                ax.xaxis.set_major_formatter(DateFormatter("%d/%b"))
                ax.xaxis.set_major_locator(MonthLocator(bymonth=list(range(1,13,3)), bymonthday=15 ))


            plt.legend(prop = FontProperties(size=8))

        if mp is None: continue
        stations_to_plot.append(s)
        plt.title("{0}: point lake fraction={1:.4f}".format(s.id, common_lake_fractions[mp.ix, mp.jy] ) )

        pp.savefig()



    #plot station positions
    plt.figure()
    bm = dmManager.get_rotpole_basemap_using_lons_lats(lons2d=dmManager.lons2D, lats2d=dmManager.lats2D, resolution="i")
    bm.drawcoastlines(linewidth=0.1)
    bm.drawrivers(linewidth=0.1)

    lons_list = [s.longitude for s in stations_to_plot]
    lats_list = [s.latitude for s in stations_to_plot]

    x_list, y_list = bm(lons_list, lats_list)
    bm.scatter(x_list, y_list, linewidths=0, s=0.5, zorder = 3)
    ax = plt.gca()
    for s, the_x, the_y in zip(stations, x_list, y_list):
        ax.annotate(s.id, xy=(the_x, the_y),xytext=(3, 3), textcoords='offset points',
            font_properties = FontProperties(size = 4), bbox = dict(facecolor = 'w', alpha = 1),
            ha = "left", va = "bottom", zorder = 2)
    pp.savefig()


    pp.close()



def dotest(sim_name_to_station_to_model_point):
    day_stamps = Station.get_stamp_days(2001)
    for sim_name, station_to_mp in sim_name_to_station_to_model_point.items():
        st_to_mp = sim_name_to_station_to_model_point[sim_name]

        for station, mp in st_to_mp.items():
            assert isinstance(mp, ModelPoint)
            years = station.get_list_of_complete_years()
            d,v = mp.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=day_stamps, years=years)
            plt.plot(d,v)
            plt.show()

            raise Exception()



def main():
    print("start")
    import application_properties
    application_properties.set_current_directory()
    from util import plot_utils

    print("finished imports and set ups")
    plot_utils.apply_plot_params(width_pt=None, width_cm=15, height_cm=15, font_size=16)
    validate_daily_climatology()

    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  