from datetime import datetime
import os
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.gridspec import GridSpec
from crcm5.model_data import Crcm5ModelDataManager
from data import cehq_station
from data.cehq_station import Station
from data.timeseries import TimeSeries
import matplotlib.pyplot as plt

__author__ = 'huziy'

import numpy as np




def diagnose(station_ids = None):
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_with_lakes"
    coord_file = os.path.join(data_path, "pm1985050100_00000000p")


    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
            file_name_prefix="pm", all_files_in_samples_folder=True
    )


    start_date = datetime(1986, 1, 1)
    end_date = datetime(1986, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = station_ids,
            start_date=start_date, end_date=end_date
    )


    fig = plt.figure()
    #two columns
    gs = GridSpec( len(stations) // 2 + len(stations) % 2, 2, hspace=0.4, wspace=0.4 )
    line_model, line_obs = None, None
    stations.sort(key=lambda x: x.latitude, reverse=True)

    for i, s in enumerate(stations):
        model_ts = manager.get_streamflow_timeseries_for_station(s, start_date = start_date, end_date = end_date)
        ax = fig.add_subplot( gs[i // 2, i % 2] )

        assert isinstance(model_ts, TimeSeries)

        #[t, m_data] = model_ts.get_daily_normals()
        #[t, s_data] = s.get_daily_normals()

        assert isinstance(s, Station)

        #climatologies
        #line_model = ax.plot(t, m_data, label = "Model (CRCM5)", lw = 3, color = "b")
        #line_obs = ax.plot(t, s_data, label = "Observation", lw = 3, color = "r")

        model_ts = model_ts.get_ts_of_daily_means()
        print model_ts.time[0], model_ts.time[-1]
        print model_ts.data[0:10]

        print model_ts.metadata

        mod_vals = model_ts.get_data_for_dates(s.dates)
        print mod_vals[:20]
        print "+" * 20
        assert len(mod_vals) == len(s.dates)

        line_model = ax.plot(s.dates, mod_vals, label = "Model (CRCM5)", lw = 1, color = "b")
        line_obs = ax.plot(s.dates, s.values, label = "Observation", lw = 3, color = "r", alpha = 0.5)

        ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod_vals, s.values])[0,1] )), xy = (0.7,0.8), xycoords= "axes fraction")


        ax.set_title("%s: da_diff=%.2f %%, dist = %.1f" % (s.id, (-s.drainage_km2+
                        model_ts.metadata["acc_area_km2"]) / s.drainage_km2 * 100.0,
                        model_ts.metadata["distance_to_obs"]))

        ax.xaxis.set_major_formatter(DateFormatter("%y/%m"))
        ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,5)))

    lines = (line_model, line_obs)
    labels = ("Model (CRCM5)", "Observation" )
    fig.legend(lines, labels)
    fig.savefig("performance_without_lakes_0.1deg_1year.png")


def main():
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                        "081006", "061502", "040830", "080718"]

    diagnose(station_ids=selected_ids)

    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  