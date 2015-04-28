from data import cehq_station
from data.cehq_station import Station

__author__ = 'huziy'


def main():

    stations = cehq_station.load_from_hydat_db(natural=True, province="SK")
    for s in stations:
        assert isinstance(s, Station)
        print("{}; {}; {}; {}; ".format(s.name, s.longitude, s.latitude, s.drainage_km2))

if __name__ == '__main__':
    main()