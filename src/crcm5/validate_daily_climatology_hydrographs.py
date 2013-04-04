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

    sim_name_list = ["crcm5-r", "crcm5-hcd-r", "crcm5-hcd-rl", "crcm5-hcd-rl-intfl"]
    rpn_folder_path_form = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup"
    nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"



    #select stations
    selected_ids = None
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )






    varname = "STFL"
    sim_name_to_manager = {}
    sim_name_to_station_to_model_point = {}
    dmManager = None



    for sim_name in sim_name_list:
        rpn_folder = rpn_folder_path_form.format(sim_name)

        dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm",
            all_files_in_samples_folder=True, need_cell_manager=True)

        sim_name_to_manager[sim_name] = dmManager

        nc_path = os.path.join(nc_db_folder, sim_name)
        nc_path = os.path.join(nc_path, "{0}_all.nc".format(varname))
        st_to_mp = dmManager.get_model_points_for_stations(stations, nc_path=nc_path, varname=varname)

        sim_name_to_station_to_model_point[sim_name] = st_to_mp


    common_lake_fractions = dmManager.lake_fraction

    #for tests
    #test(sim_name_to_station_to_model_point)




    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('comp_with_obs_daily_clim.pdf')

    day_stamps = Station.get_stamp_days(2001)
    for s in stations:
        plt.figure()

        assert isinstance(s, Station)

        years = s.get_list_of_complete_years()

        print s
        if len(years) < 6: continue

        dates, obs_data = s.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=day_stamps, years=years)
        obs_ann_mean = np.mean(obs_data)

        plt.plot( dates, obs_data, label = "Obs: ann.mean = {0:.1f}".format( obs_ann_mean ) )

        mp = None
        for sim_name in sim_name_list:
            manager = sim_name_to_manager[sim_name]
            if not sim_name_to_station_to_model_point[sim_name].has_key(s):
                continue

            mp = sim_name_to_station_to_model_point[sim_name][s]
            assert isinstance(mp, ModelPoint)
            dates, values = mp.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=day_stamps, years=years)
            plt.plot(dates, values , label = "{0}: {1:.2f} \n ann.mean = {2:.1f}".format( sim_name,
                manager.lake_fraction[mp.flow_in_mask == 1].mean(), np.mean(values)) )

            ax = plt.gca()
            assert isinstance(ax, Axes)

            ax.xaxis.set_major_formatter(DateFormatter("%b/%d"))
            ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,3) ))


            plt.legend(prop = FontProperties(size=8))

        if mp is None: continue
        plt.title("{0}: point lake fraction={1:.4f}".format(s.id, common_lake_fractions[mp.ix, mp.jy] ) )

        pp.savefig()


    pp.close()



def test(sim_name_to_station_to_model_point):
    day_stamps = Station.get_stamp_days(2001)
    for sim_name, station_to_mp in sim_name_to_station_to_model_point.iteritems():
        st_to_mp = sim_name_to_station_to_model_point[sim_name]

        for station, mp in st_to_mp.iteritems():
            assert isinstance(mp, ModelPoint)
            years = station.get_list_of_complete_years()
            d,v = mp.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=day_stamps, years=years)
            plt.plot(d,v)
            plt.show()

            raise Exception()



def main():
    import application_properties
    from util import plot_utils

    plot_utils.apply_plot_params(width_pt=None, width_cm=15, height_cm=20, font_size=16)
    application_properties.set_current_directory()
    validate_daily_climatology()

    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  