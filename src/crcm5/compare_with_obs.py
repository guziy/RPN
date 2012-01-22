from datetime import datetime
import application_properties
from data import cehq_station

__author__ = 'huziy'

import numpy as np

#Compares modelled and observed hydrographs




def main():
    start_date = datetime(1985,1,1)
    end_date = datetime(1990,12, 31)
    stations = cehq_station.read_station_data(start_date = start_date, end_date=end_date)
    print "Read {0} stations".format(len(stations))
    for s in stations:
        if not len(s.dates):
            print s.id, "=>", len(s.dates), "from: -- to -- "
            continue
        print s.id, "=>", len(s.dates), "from: ", s.dates[0], " to ", s.dates[-1]
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  