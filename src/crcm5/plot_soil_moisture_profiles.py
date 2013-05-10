from datetime import datetime
from crcm5.model_data import Crcm5ModelDataManager
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.cell_manager import CellManager
import matplotlib.pyplot as plt


__author__ = 'huziy'

import numpy as np

#select a region on a map, and show 12 profiles (- climatological mean for each month)
# I1 - soil moisture
#
# Given a station calculate mean profile over upstream model points




def main():

    #peirod of interest
    start_year = 1979
    end_year = 1988

    #spatial averaging will be done over upstream points to the stations
    selected_ids = ["093801"]

    #simulation names corresponding to the paths
    sim_names = ["crcm5-hcd-rl", "crcm5-hcd-rl-intfl"]

    sim_labels = [x.upper() for x in sim_names]

    colors = ["blue", "violet"]

    paths = [
        "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl_spinup",
        "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup2/Samples_all_in_one"

    ]


    seasons = [
        [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]
    ]

    managers = [
        Crcm5ModelDataManager(samples_folder_path=path, file_name_prefix="pm",
            all_files_in_samples_folder=True, need_cell_manager= (i == 0)) for i, path in enumerate(paths)
    ]

    #share the cell manager
    a_data_manager = managers[0]
    assert isinstance(a_data_manager, Crcm5ModelDataManager)
    cell_manager = a_data_manager.cell_manager
    assert isinstance(cell_manager, CellManager)
    for m in managers[1:]:
        assert isinstance(m, Crcm5ModelDataManager)
        m.cell_manager = cell_manager

    #share the lake fraction field
    lake_fraction = a_data_manager.lake_fraction



    #selected_ids = ["092715", "080101", "074903", "050304", "080104", "081007", "061905",
    #                  "041903", "040830", "093806", "090613", "081002", "093801", "080718"]
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )

    #stations with corresponding model points
    station_to_mp = a_data_manager.get_dataless_model_points_for_stations(stations)

    #figure out levels in soil


    a_data_manager.get_monthly_climatology_of_3d_field(var_name="I1", start_year=start_year, end_year=end_year)

    if True: return

    for s, mp in station_to_mp.iteritems():
        assert isinstance(mp, ModelPoint)
        mask = (mp.flow_in_mask == 1) & (lake_fraction < 0.6)
        fig = plt.figure()

        for m, label, color in zip(managers, sim_labels, colors):
            #a_data_manager.get_mean_field()
            pass




    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  