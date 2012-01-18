import application_properties
from data import cehq_station

__author__ = 'huziy'

import numpy as np

#Compares modelled and observed hydrographs




def main():
    stations = cehq_station.read_station_data()
    print "Read {0} stations".format(len(stations))
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  