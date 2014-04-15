from datetime import datetime
import pickle
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, date2num
from matplotlib.ticker import ScalarFormatter
import os
from crcm5.model_data import Crcm5ModelDataManager
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.cell_manager import CellManager
import matplotlib.pyplot as plt
import my_colormaps


__author__ = 'huziy'

import numpy as np

#select a region on a map, and show 12 profiles (- climatological mean for each month)
# I1 - soil moisture
#
# Given a station calculate mean profile over upstream model points


def _get_cache_file_name(sim_name, start_year, end_year, var_name):
    return "{0}_{1}_{2}_{3}_cache_3d.bin".format(sim_name, start_year, end_year, var_name)


def _get_cached_monthly_mean_fields(sim_name, start_year, end_year, var_name):
    fName = _get_cache_file_name(sim_name, start_year, end_year, var_name)
    if os.path.isfile(fName):
        return pickle.load(open(fName))
    return None


def _cache_monthly_mean_fields(data, sim_name, start_year, end_year, var_name):
    fName = _get_cache_file_name(sim_name, start_year, end_year, var_name)
    pickle.dump(data, open(fName, mode="w"))


def plot_at_indices(ix,jy):
    var_name_liquid = "I1"
    var_name_solid = "I2"
    #peirod of interest
    start_year = 1979
    end_year = 1988



    #simulation names corresponding to the paths
    sim_names = ["crcm5-hcd-rl", "crcm5-hcd-rl-intfl"]

    sim_labels = [x.upper() for x in sim_names]


    layer_widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 3.0, 5.0,
                    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    layer_depths = np.cumsum(layer_widths)


    paths = [
        "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl_spinup",
        "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup2/Samples_all_in_one"
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



    selected_ids = ["092715", "080101", "074903", "050304", "080104", "081007", "061905",
                      "041903", "040830", "093806", "090613", "081002", "093801", "080718"]
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )

    #stations with corresponding model points
    station_to_mp = a_data_manager.get_dataless_model_points_for_stations(stations)

    #figure out levels in soil



    sim_label_to_profiles = {}
    fig = plt.figure()
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits([-2, 2])

    for m, label in zip(managers, sim_labels):
        assert isinstance(m, Crcm5ModelDataManager)

        monthly_means_liquid = _get_cached_monthly_mean_fields(label, start_year, end_year, var_name_liquid)
        if monthly_means_liquid is None:
            monthly_means_liquid = m.get_monthly_climatology_of_3d_field(var_name=var_name_liquid, start_year=start_year
                , end_year=end_year)
            _cache_monthly_mean_fields(monthly_means_liquid, label, start_year, end_year, var_name_liquid)

        monthly_means_solid = _get_cached_monthly_mean_fields(label, start_year, end_year, var_name_solid)
        if monthly_means_solid is None:
            monthly_means_solid = m.get_monthly_climatology_of_3d_field(var_name=var_name_solid, start_year=start_year,
                end_year=end_year)
            _cache_monthly_mean_fields(monthly_means_solid, label, start_year, end_year, var_name_solid)

        profiles = [monthly_means_liquid[i][ix,jy, :] + monthly_means_solid[i][ix, jy, :] for i
                    in range(12)]

        sim_label_to_profiles[label] = np.array(profiles)

    x = range(12)
    y = layer_depths

    y2d, x2d = np.meshgrid(y, x)
    plt.contourf(x2d, y2d, sim_label_to_profiles[sim_labels[1]] - sim_label_to_profiles[sim_labels[0]])
    plt.gca().invert_yaxis()
    plt.colorbar()

    #fig.tight_layout()

    fig.savefig("soil_profile_at_ix={0};jy={1}.pdf".format(ix, jy))


def main():
    var_name_liquid = "I1"
    var_name_solid = "I2"
    #peirod of interest
    start_year = 1979
    end_year = 1988

    #spatial averaging will be done over upstream points to the stations
    selected_ids = ["092715", "080101", "074903", "050304", "080104", "081007", "061905",
                         "041903", "040830", "093806", "090613", "081002", "093801", "080718"]

    selected_ids = ["074903"]


    #simulation names corresponding to the paths
    sim_names = ["crcm5-hcd-rl", "crcm5-hcd-rl-intfl"]

    sim_labels = [x.upper() for x in sim_names]

    colors = ["blue", "violet"]

    layer_widths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 3.0, 5.0,
                    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    layer_depths = np.cumsum(layer_widths)


    paths = [
        "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl_spinup",
        "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl-intfl_spinup2/Samples_all_in_one"
    ]

    seasons = [
        [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]
    ]

    season_names = [
        "DJF", "MAM", "JJA", "SON"
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



    sim_label_to_profiles = {}
    for s, mp in station_to_mp.iteritems():
        assert isinstance(mp, ModelPoint)
        mask = (mp.flow_in_mask == 1) & (lake_fraction < 0.6)
        fig = plt.figure()
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits([-2, 2])

        print mp.ix, mp.jy, s.id

        for m, label, color in zip(managers, sim_labels, colors):
            assert isinstance(m, Crcm5ModelDataManager)



            monthly_means_liquid = _get_cached_monthly_mean_fields(label, start_year, end_year, var_name_liquid)
            if monthly_means_liquid is None:
                monthly_means_liquid = m.get_monthly_climatology_of_3d_field(var_name=var_name_liquid, start_year=start_year, end_year=end_year)
                _cache_monthly_mean_fields(monthly_means_liquid, label, start_year, end_year, var_name_liquid)

            monthly_means_solid = _get_cached_monthly_mean_fields(label, start_year, end_year, var_name_solid)
            if monthly_means_solid is None:
                monthly_means_solid = m.get_monthly_climatology_of_3d_field(var_name=var_name_solid, start_year=start_year, end_year=end_year)
                _cache_monthly_mean_fields(monthly_means_solid, label, start_year, end_year, var_name_solid)


            profiles = [ monthly_means_liquid[i][mask,:].mean(axis = 0) + monthly_means_solid[i][mask,:].mean(axis = 0) for i in range(12) ]

            sim_label_to_profiles[label] = np.array( profiles )


        x = [ date2num( datetime(2001,month,1) ) for month in range(1,13)]
        y = layer_depths

        y2d, x2d = np.meshgrid(y, x)
        delta = (sim_label_to_profiles[sim_labels[1]] - sim_label_to_profiles[sim_labels[0]]) / sim_label_to_profiles[sim_labels[0]] * 100

        #delta = np.ma.masked_where(delta < 0.1, delta)

        cmap = my_colormaps.get_cmap_from_ncl_spec_file(path="colormap_files/BlueRed.rgb", ncolors=10)
        the_min = -6.0
        the_max = 6.0
        step = (the_max - the_min) / float(cmap.N)

        plt.pcolormesh(x2d[:,:8], y2d[:,:8], delta[:,:8], cmap = cmap, vmin = the_min, vmax = the_max) #, levels = np.arange(-6,7,1))
        plt.gca().invert_yaxis()
        plt.colorbar(ticks = np.arange(the_min, the_max + step, step))
        plt.gca().set_ylabel("Depth (m)")

        plt.gca().xaxis.set_major_formatter(DateFormatter("%b"))


        #fig.tight_layout()
        fig.savefig("soil_profile_upstream_of_{0}.pdf".format(s.id))




    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=30, height_cm=15, font_size=20)
    main()
    print "Hello world"
  