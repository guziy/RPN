__author__ = 'huziy'

import numpy as np
from data import cehq_station as cehq

def main():

    lake_names = ["Unnamed", "Mistassini", "Simon", "Nemiscan"]
    level_st_ids = [
        "080716", "081003", "040408", "081001"
    ]
    level_stations = cehq.read_station_data(folder="data/cehq_levels",
                        selected_ids=level_st_ids)

    stfl_st_ids = [
        "080701", "081007", "040401", "081002"
    ]
    stfl_stations = cehq.read_station_data(folder="data/cehq_measure_data")




    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  