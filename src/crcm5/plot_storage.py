from datetime import datetime
import itertools
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from numpy.lib.function_base import meshgrid
from scipy.spatial.kdtree import KDTree
import application_properties
from crcm5.model_data import Crcm5ModelDataManager
from data import cehq_station
from data.cehq_station import Station
from data.timeseries import DateValuePair, TimeSeries
from permafrost import draw_regions
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from rpn.rpn import RPN
import os

import matplotlib.pyplot as plt

def main():
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_with_lakes"


    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_198501_198612_0.1deg"
    coord_file = os.path.join(data_path, "pm1985010100_00000000p")


    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
            file_name_prefix="pm", all_files_in_samples_folder=True
    )

    assert isinstance(manager, Crcm5ModelDataManager)
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                    "081006", "061502", "040830", "080718"]

    start_date = datetime(1985, 1, 1)
    end_date = datetime(1985, 12, 31)

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

        model_ts = manager.get_streamflow_timeseries_for_station(s, start_date = start_date,
            end_date = end_date, var_name="SWSR")
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

        line_model = ax.plot(s.dates, mod_vals, label = "Model (CRCM5)", lw = 1, color = "b")
        #line_obs = ax.plot(s.dates, s.values, label = "Observation", lw = 3, color = "r", alpha = 0.5)

        bf_store = model_ts.metadata["bankfull_store_m3"]
        ax.plot([s.dates[0], s.dates[-1]], [bf_store, bf_store], color = "k")

        ax.set_title("%s: da_diff=%.2f %%, dist = %.1f" % (s.id, (-s.drainage_km2+
                        model_ts.metadata["acc_area_km2"]) / s.drainage_km2 * 100.0,
                        model_ts.metadata["distance_to_obs"]))

        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_major_locator(YearLocator())

    fig.savefig("storage.png")


    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  