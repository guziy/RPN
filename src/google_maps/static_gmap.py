__author__ = 'huziy'

import numpy as np

import http.client
import urllib.request, urllib.parse, urllib.error
import application_properties
from data import cehq_station
def main():
    #url = "http://maps.googleapis.com/maps/api/staticmap?center=46.5,-72&zoom=15&size=800x800&maptype=satellite&sensor=false"
    params = {
        "center" : "46.5,-72",
        "zoom" : "5",
        "size" : "800x800",
        "maptype" : "satellite",
        "sensor" : "false"
    }
    script = "http://maps.googleapis.com/maps/api/staticmap?"
    print(urllib.parse.urlencode(params))
    url = script + urllib.parse.urlencode(params)


    stations = cehq_station.read_station_data(folder="data/cehq_levels")[:5]
    for s in stations:
        assert isinstance(s, cehq_station.Station)
        url += "&" + urllib.parse.urlencode({"markers": "color:red|label:%s|%f,%f" % (s.id, s.latitude, s.longitude)})


    print(url)
    urllib.request.urlretrieve(url, filename="gmap.png")



    #TODO: implement
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print("Hello world")
