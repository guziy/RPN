from datetime import datetime
import os
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
from matplotlib.gridspec import GridSpec
from crcm5.model_data import Crcm5ModelDataManager
from data import cehq_station
from data.cehq_station import Station
from data.timeseries import TimeSeries
from util import plot_utils
import matplotlib.pyplot as plt
__author__ = 'huziy'

import numpy as np

def main():
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_198501_198612_0.1deg"
    coord_file = os.path.join(data_path, "pm1985010100_00000000p")


    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
            file_name_prefix="pm", all_files_in_samples_folder=True
    )
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                    "081006", "061502", "040830", "080718"]

    start_date = datetime(1986, 1, 1)
    end_date = datetime(1986, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )
    plot_utils.apply_plot_params(width_pt=None, height_cm =30.0, width_cm=16, font_size=10)
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

        mod_vals = model_ts.get_data_for_dates(s.dates)
        print mod_vals[:20]
        print "+" * 20
        assert len(mod_vals) == len(s.dates)

        #model_acorr = [1] + [ np.corrcoef([mod_vals[i:], mod_vals[:-i]])[0,1] for i in range(1,len(mod_vals)) ]
        #obs_acorr = [1] + [ np.corrcoef([s.values[i:], s.values[:-i]])[0,1] for i in range(1,len(mod_vals)) ]



        npoints = np.array(range(len(mod_vals), 0, -1))

        model_acorr = np.correlate(mod_vals, mod_vals, mode="full")
        model_acorr = model_acorr[len(model_acorr) / 2:] / max(model_acorr)
        model_acorr /= npoints

        obs_acorr = np.correlate(s.values, s.values, mode = "full")
        obs_acorr = obs_acorr[len(obs_acorr) / 2 :] / max(obs_acorr)
        obs_acorr /= npoints

        print len(model_acorr), len(s.dates)

        line_model = ax.plot(s.dates, model_acorr, label = "Model (CRCM5)", lw = 1, color = "b")
        line_obs = ax.plot(s.dates, obs_acorr, label = "Observation", lw = 3, color = "r", alpha = 0.5)

        #ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod_vals, s.values])[0,1] )), xy = (0.7,0.8), xycoords= "axes fraction")


        ax.set_title("%s: da_diff=%.2f %%, dist = %.1f" % (s.id, (-s.drainage_km2+
                        model_ts.metadata["acc_area_km2"]) / s.drainage_km2 * 100.0,
                        model_ts.metadata["distance_to_obs"]))

        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13, 2)))

    lines = (line_model, line_obs)
    labels = ("Model (CRCM5)", "Observation" )
    fig.legend(lines, labels)
    fig.savefig("acorr_without_lakes_0.1deg_1year.png")

    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  