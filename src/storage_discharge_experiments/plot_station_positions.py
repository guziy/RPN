import itertools
from data.cehq_station import Station
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from data import cehq_station as cehq
import matplotlib.pyplot as plt


def get_active_storages(h_list, area_km2):
    """
    returns the list of active storages in m**3
    Si = (Hi - min(Hi)) * area_km2
    """
    h_min = min(h_list) * 0.6
    return map(lambda x: (x - h_min) * area_km2 * 1e6, h_list)


def get_streamflows_from_active_stores(stores, area_km2, kd = 0.01 / (24.0*60.0*60.0)):
    """
    Q = S * (S/S0)**1.5 * kd
    """
    h0 = 5.0
    s0 = h0 * area_km2 * 1.0e6
    return map(lambda x: x * (x / s0) ** 1.5 * kd, stores )
    pass





def main():

    lake_names = ["Matagami", "Mistassini", "Nemiscau"]
    lake_areas_km2 = [370.7,  2162.7, 148.3]

    level_st_ids = [
        "080716", "081003",  "081001"
    ]
    level_stations = cehq.read_station_data(folder="data/cehq_levels",
                        selected_ids=level_st_ids)

    stfl_st_ids = [
        "080707", "081007", "081002"
    ]
    stfl_stations = cehq.read_station_data(folder="data/cehq_measure_data",
            selected_ids=stfl_st_ids)

    q_obs = []
    q_calc = []

    plt.figure()
    for lev_station, stfl_station, lake_name, lake_area_km2, c in zip(level_stations,
                                                                    stfl_stations, lake_names, lake_areas_km2,
                                                                    ["r","g","b"]
                                                                ):
        assert isinstance(lev_station, Station)
        assert isinstance(stfl_station, Station)

        count_intersection = sum( map(lambda d: int(d in stfl_station.dates), lev_station.dates) )
        intersection_dates = list( itertools.ifilter( lambda d: d in stfl_station.dates, lev_station.dates) )

        q_vals = map( lambda d: stfl_station.get_value_for_date(d), intersection_dates )
        h_vals = map( lambda d: lev_station.get_value_for_date(d), intersection_dates )


        q_obs.append(q_vals)
        q_calc.append(get_streamflows_from_active_stores(get_active_storages(h_vals,lake_area_km2), lake_area_km2))

        #Calculate correlation between Q and H
        print 10 * "-" + lake_name
        print "r = {0}".format(np.corrcoef([q_vals, h_vals])[0,1])
        print(lev_station.latitude, lev_station.longitude)

        print "dist_m = {0} km ".format( 1.0e-3 * lat_lon.get_distance_in_meters(lev_station.longitude, lev_station.latitude,
                                                                                stfl_station.longitude, stfl_station.latitude))



        print "{0} and {1} have {2} measurements at the same time ({3}).".format(lev_station.id, stfl_station.id,
            count_intersection, lake_name )

        #plot storage-discharge relation
        plt.title(lake_name)
        plt.scatter(get_active_storages(h_vals,lake_area_km2), q_vals, c = c , label = lake_name, linewidths=0)


        #plt.plot(intersection_dates, q_vals, c, label = lake_name )
        #plt.plot(intersection_dates, get_active_storages(h_vals,lake_area_km2), c+"-", label = lake_name )


        #plt.xlabel("S-active (obs)")
        #plt.ylabel("Q (obs)")
    plt.legend()


    #Compare observed and theoretical lake outflows
    plt.figure()
    title = ""

    for qo, qc, name, c  in zip(q_obs, q_calc,lake_names, ["r","g","b"]):
        qc_a = np.array(qc)
        qo_a = np.array(qo)
        title = "ME = {0:.2f}".format(1 - sum((qc_a - qo_a) ** 2) / sum( (qo_a - qo_a.mean()) ** 2))
        plt.scatter(qo, qc, c= c, linewidths=0,label = name + ", " + title)

    #plt.title(title)
    plt.xlabel("$Q_{\\rm obs}$")
    plt.ylabel("$Q_{\\rm mod}$")

    xmin, xmax = plt.xlim()
    print plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], "k-",lw = 3, zorder = 5)

    plt.legend()
    plt.show()


    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    main()
    print "Hello world"
  