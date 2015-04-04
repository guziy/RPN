from datetime import datetime
import itertools
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from data.cehq_station import Station
from util import plot_utils
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
    h_min = max( min(h_list) - 5, 0)
    if h_min == 0.1:
        print(h_min)
    return [(x - h_min) * area_km2 * 1e6 for x in h_list]


def get_streamflows_from_active_stores(stores, area_km2, kd = 0.01 / (24.0*60.0*60.0)):
    """
    Q = S * (S/S0)**1.5 * kd
    """
    h0 = 5.0
    s0 = h0 * area_km2 * 1.0e6
    return [x * (x / s0) ** 1.5 * kd for x in stores]
    pass


def get_streamflows_from_active_stores_bowling(stores, area_km2):
    """
    See Bowling et al 2009 for the formula explanation
    """
    cd = 0.94
    f = 0.015
    g = 9.81

    area_m2 = area_km2 * 1.0e6
    return [( 4.0 / 3.0 ) * cd * f * (g * np.pi) ** 0.5 * s ** 1.5 / area_m2 for s in stores]



def get_ice_factor(q, s, qb, sb):
    return (q/qb) / (s/sb) ** 2.5



def plot_for_different_months(start_date = None, end_date = None):
    lake_names = [
        #"Matagami",
        "Mistassini", "Nemiscau"]
    lake_areas_km2 = [
        #370.7,
        2162.7, 148.3]

    level_st_ids = [
        #"080716",
        "081003",  "081001"
    ]
    level_stations = cehq.read_station_data(folder="data/cehq_levels",
                        selected_ids=level_st_ids)

    stfl_st_ids = [
        #"080707",
        "081007", "081002"
    ]
    stfl_stations = cehq.read_station_data(folder="data/cehq_measure_data",
            selected_ids=stfl_st_ids)


    for lev_station, stfl_station, lake_name, lake_area_km2, c in zip(level_stations,
                                                                    stfl_stations, lake_names, lake_areas_km2,
                                                                    ["r","g","b"]
                                                                ):
        assert isinstance(lev_station, Station)
        assert isinstance(stfl_station, Station)

        all_q_obs = []
        all_q_calc = []
        all_k = []
        all_b = []


        intersection_dates = list( sorted( filter( lambda d: d in stfl_station.dates, lev_station.dates) ) )

        q_vals = [stfl_station.get_value_for_date(d) for d in intersection_dates]
        h_vals = [lev_station.get_value_for_date(d) for d in intersection_dates]
        #change the way streamflow calculated here
        q_calc = get_streamflows_from_active_stores_bowling(get_active_storages(h_vals,lake_area_km2), lake_area_km2)


        q_min = min( min(q_vals), min(q_calc) )
        q_max = max( max(q_vals), max(q_calc) )

        plot_utils.apply_plot_params(width_pt=None, width_cm=30, height_cm=20, font_size=10)
        fig = plt.figure()
        gs = gridspec.GridSpec(3,4,wspace=0.5, hspace=0.5)

        if start_date is None:
            start_date = intersection_dates[0]

        if end_date is None:
            end_date = intersection_dates[-1]
        print("lake name: {0}".format(lake_name))
        print("data are from " + str( intersection_dates[0] ) + " to " + str( intersection_dates[-1] ))


        ##As a base month we are using August (assuming that we don't have ice in August)
        base_month = 8
        the_dates = list( filter( lambda d: (start_date <= d <= end_date) and
                                                        d.month == base_month, intersection_dates) )
        q_base = np.mean([stfl_station.get_value_for_date(d) for d in the_dates])
        h_vals = [lev_station.get_value_for_date(d) for d in the_dates]
        store_base = np.mean(get_active_storages(h_vals,lake_area_km2))

        ice_factors = []
        for month in range(1, 13):
            ax = fig.add_subplot(gs[(month - 1)//4, (month - 1) % 4])

            the_dates = list( filter( lambda d: (start_date <= d <= end_date) and
                                                           d.month == month, intersection_dates) )

            q_vals = [stfl_station.get_value_for_date(d) for d in the_dates]
            h_vals = [lev_station.get_value_for_date(d) for d in the_dates]
            s_vals = get_active_storages(h_vals,lake_area_km2)
            print(len(h_vals))
            q_calc = get_streamflows_from_active_stores_bowling(s_vals, lake_area_km2)


            ice_factors.append(get_ice_factor(np.mean(q_vals), np.mean(s_vals), q_base, store_base))


            all_q_obs.extend(q_vals)
            all_q_calc.extend(q_calc)

            ax.scatter(q_vals, q_calc, linewidths = 0)

            print("len(q_vals) = {0}".format(len(q_vals)))

            ax.set_xlabel("$Q_{\\rm obs}$")
            ax.set_ylabel("$Q_{\\rm mod}$")

            the_poly = np.polyfit(q_vals, q_calc, 1)
            k, b = the_poly

            all_k.append(k)
            all_b.append(b)

            ax.scatter(q_vals, [(x - b) / k for x in q_calc], c ="r", linewidth = 0, zorder = 6)




            #ax.annotate("k={0:.2f}; \nb={1:d}".format(k, int(b)), xy = (0.6, 0.05),
            #    xycoords = "axes fraction", zorder = 5
            #)
            ax.plot([q_min, q_max], [q_min, q_max], "k-",lw = 3, zorder = 5)
            d = datetime(2000, month, 1)
            ax.set_title(d.strftime("%b") + "( k={0:.7f}; b={1:.2f})".format(k, b))
            ax.xaxis.set_major_locator(MultipleLocator(base = np.round((q_max - q_min )/ 10) * 10 / 2  ))
            ax.yaxis.set_major_locator(MultipleLocator(base = np.round((q_max - q_min ) / 10) * 10 / 2))

        fig.suptitle(lake_name)
        fig.savefig("{0}.png".format(lake_name))
        assert isinstance(fig, Figure)


        plot_utils.apply_plot_params(width_pt=None, width_cm=30, height_cm=20, font_size=25)
        #plot k and b as a function of month
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        assert isinstance(ax, Axes)
        ax.set_title("Correction coefficients applied \n q'=(1/k)*q-b/k")
        ax.plot(list(range(1,13)), 1.0 / np.array(all_k), lw = 3, label = "1/k")
        ax.legend()
        fig.savefig("{0}_coefs_1_k.png".format(lake_name))

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        assert isinstance(ax, Axes)
        ax.set_title("Correction coefficients applied \n q'=(1/k)*q-b/k")
        ax.plot(list(range(1,13)), -np.array(all_b) / np.array(all_k), lw = 3, label = "-b/k")
        ax.legend()
        fig.savefig("{0}_coefs_b_k.png".format(lake_name))


        plt.figure()
        plt.plot(list(range(1,13)), ice_factors, lw = 3)
        plt.title("Ice factor for {0}".format(lake_name))
        plt.savefig("{0}_ice_factors.png".format(lake_name))


        ###plot q-q for all season in the same plot
        plot_utils.apply_plot_params(width_pt=None, width_cm=30, height_cm=20, font_size=20)
        fig1 = plt.figure()
        ax = fig1.add_subplot(1,1,1)
        ax.set_title(lake_name)
        ax.scatter(all_q_obs, all_q_calc, c="b", linewidths=0)
        the_poly = np.polyfit(all_q_obs, all_q_calc, 1)
        k,b = the_poly

        print(k, b)
        print("len(all_q) = {0}".format(len(all_q_calc)))

        ax.scatter(all_q_obs, [(x - b) / k for x in all_q_calc], c ="r",
            linewidth = 0, zorder = 6)
        ax.annotate("k={0:.2f}; \nb={1:.2f}".format(k, b), xy = (0.6, 0.05),
                        xycoords = "axes fraction", zorder = 7
                    )

        print(np.polyfit(all_q_obs, [(x - b) / k for x in all_q_calc],1))

        print("min(all_q_calc) = {0}, max(all_q_calc) = {1}".format(min(all_q_calc), max(all_q_calc)))


#        ax.plot([q_min, q_max], [k * q_min + b, k * q_max + b], "g" )

        ax.plot([q_min, q_max], [q_min, q_max], "k-",lw = 1, zorder = 5)

#        for the_k, the_b in zip(all_k, all_b):
#            ax.plot([q_min, q_max], [the_k * q_min + the_b, the_k * q_max + the_b] )


        ax.grid(b = True)
        ax.set_xlabel("$Q_{\\rm obs}$")
        ax.set_ylabel("$Q_{\\rm mod}$")

        ax.xaxis.set_major_locator(MultipleLocator(base = np.round((q_max - q_min )/ 10) * 10 / 2  ))
        ax.yaxis.set_major_locator(MultipleLocator(base = np.round((q_max - q_min ) / 10) * 10 / 2))

        print("all_{0}.png".format(lake_name))
        #plt.show()
        fig1.subplots_adjust(left = 0.2)
        fig1.savefig("all_{0}.png".format(lake_name))




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

        count_intersection = sum( [int(d in stfl_station.dates) for d in lev_station.dates] )
        intersection_dates = list( filter( lambda d: d in stfl_station.dates, lev_station.dates) )

        q_vals = [stfl_station.get_value_for_date(d) for d in intersection_dates]
        h_vals = [lev_station.get_value_for_date(d) for d in intersection_dates]


        q_obs.append(q_vals)
        q_calc.append(get_streamflows_from_active_stores_bowling(get_active_storages(h_vals,lake_area_km2), lake_area_km2))

        #Calculate correlation between Q and H
        print(10 * "-" + lake_name)
        print("r = {0}".format(np.corrcoef([q_vals, h_vals])[0,1]))
        print((lev_station.latitude, lev_station.longitude))

        print("dist_m = {0} km ".format( 1.0e-3 * lat_lon.get_distance_in_meters(lev_station.longitude, lev_station.latitude,
                                                                                stfl_station.longitude, stfl_station.latitude)))



        print("{0} and {1} have {2} measurements at the same time ({3}).".format(lev_station.id, stfl_station.id,
            count_intersection, lake_name ))

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
    print(plt.xlim())
    plt.plot([xmin, xmax], [xmin, xmax], "k-",lw = 3, zorder = 5)


    plt.legend()
    plt.show()


    #TODO: implement
    pass

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    #main()
    plot_utils.apply_plot_params(width_pt=None, width_cm=30, height_cm=20, font_size=10)
    plot_for_different_months()
    print("Hello world")
  