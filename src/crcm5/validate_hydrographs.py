from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter
import numpy as np
from crcm5.model_point import ModelPoint

print np.__version__

from datetime import datetime
import os
from crcm5.model_data import Crcm5ModelDataManager
from data import cehq_station
from data.cehq_station import Station
from util.geo import lat_lon

__author__ = 'huziy'


import matplotlib.pyplot as plt


def regenerate_station_to_gridcell_mapping(start_year, end_year, model_manager):
    """
    should be called when grid or search algorithm change
    """



    assert isinstance(model_manager, Crcm5ModelDataManager)


    ktree = model_manager.kdtree
    model_acc_area = model_manager.accumulation_area_km2
    model_acc_area_1d = model_acc_area.flatten()


#    selected_ids = ["104001", "103715",
#                    "093806", "093801",
#                    "092715",
#                    "081006", "040830"]

    selected_ids = None
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )


    station_to_grid_point = {}
    for s in stations:
        assert isinstance(s, Station)
        x, y, z = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)
        dists, inds = ktree.query((x,y,z), k = 8)


        deltaDaMin = np.min( np.abs( model_acc_area_1d[inds] - s.drainage_km2) )

        imin = np.where(np.abs( model_acc_area_1d[inds] - s.drainage_km2) == deltaDaMin)[0]

        deltaDa2D = np.abs(model_acc_area - s.drainage_km2)

        ij = np.where(  deltaDa2D == deltaDaMin )

        mp = ModelPoint()
        mp.accumulation_area = model_acc_area[ij[0][0], ij[1][0]]
        mp.ix = ij[0][0]
        mp.jy = ij[1][0]
        mp.longitude = model_manager.lons2D[mp.ix, mp.jy]
        mp.latitude = model_manager.lats2D[mp.ix, mp.jy]

        #flow in mask
        mp.flow_in_mask = model_manager.get_mask_for_cells_upstream(mp.ix, mp.jy)

        station_to_grid_point[s] = mp

        print "da_diff (sel) = ", deltaDaMin
        print "dist (sel) = ", dists[imin]

    return station_to_grid_point


def validate_as_is():
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




    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('comp_with_obs_as_is.pdf')
    for s in stations:

        #check the availability of the data
        assert isinstance(s, Station)
        #s.get_longest_continuous_series()

        plt.figure()
        obs_data = [s.date_to_value[d] for d in s.dates ]
        obs_ann_mean = np.mean(obs_data)
        plt.plot( s.dates,  [s.date_to_value[d] for d in s.dates ], label = "Obs \n ann.mean = {0:.1f}".format(obs_ann_mean) )

        mp = None
        for sim_name in sim_name_list:
            manager = sim_name_to_manager[sim_name]
            if not sim_name_to_station_to_model_point[sim_name].has_key(s):
                continue

            mp = sim_name_to_station_to_model_point[sim_name][s]
            plt.plot(mp.time, mp.data[:, 0] , label = "{0}: {1:.2f} \n ann.mean = {2:.1f}".format( sim_name,
                manager.lake_fraction[mp.flow_in_mask == 1].mean(), mp.data[:,0].mean()) )
            plt.legend()

        if mp is None: continue
        plt.title("{0}: point lake fraction={1:.4f}".format(s.id, common_lake_fractions[mp.ix, mp.jy] ) )

        pp.savefig()



    pp.close()



def main():




    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    #validate_as_is()

    validate_daily_climatology()
    print "Hello world"
  