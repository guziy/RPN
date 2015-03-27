import os

__author__ = 'san'

import numpy as np

def to_projection(lon, lat, r_earth=6371e3):
    lon_r, lat_r = [np.radians(c) for c in (lon, lat)]
    return lon_r * r_earth * np.cos(lat_r), lat_r * r_earth


def print_header(path="~/NEMO/validation/glerl_ice_data/g20030403.ct"):
    path = os.path.expanduser(path)
    with open(path) as f:
        for i in range(6):
            print f.next()


def print_min_lon_and_lat():
    path="~/NEMO/validation/glerl_ice_data/Longrid.txt"
    arr_lon = np.loadtxt(os.path.expanduser(path))

    path="~/NEMO/validation/glerl_ice_data/Latgrid.txt"
    arr_lat = np.loadtxt(os.path.expanduser(path))

    print "min lon: {}".format(arr_lon.min())
    print "min lat: {}".format(arr_lat.min())


def main():
    print_min_lon_and_lat()
    lon = -92.36612
    lat = 38.84815
    print to_projection(lon=lon, lat=lat)

    print_header()


if __name__ == '__main__':
    main()
