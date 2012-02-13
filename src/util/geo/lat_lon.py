__author__="huziy"
__date__ ="$13 juil. 2010 13:34:52$"

from math import atan2
from util.geo.GeoPoint import GeoPoint
import numpy as np

EARTH_RADIUS_METERS = 6371000.0

#longitude and latitude are in radians
def get_nvector(rad_lon, rad_lat):
    return [ np.cos(rad_lat) * np.cos(rad_lon), np.cos(rad_lat) * np.sin(rad_lon), np.sin(rad_lat) ]


#p1 and p2 are geopoint objects
def get_distance_in_meters(*arg):
    """
    arg = point1, point2
    arg = lon1, lat1, lon2, lat2
    """
    if len(arg) == 2: #if we have 2 geoppoints as an argument
        [p1, p2] = arg
        n1 = p1.get_nvector()
        n2 = p2.get_nvector()
    elif len(arg) == 4: #if we have the coordinates of two points in degrees
        [lon1, lat1, lon2, lat2] = np.radians(arg)
        n1 = get_nvector(lon1, lat1)
        n2 = get_nvector(lon2, lat2)
    else:
        raise Exception("get_distance_in_meters should be 2 or 4 parameters.")
    return EARTH_RADIUS_METERS * get_angle_between_vectors(n1, n2)


def get_angle_between_vectors(n1, n2):
    dy = np.cross(n1, n2)
    dy = np.dot(dy, dy) ** 0.5
    dx = np.dot(n1, n2)
    return atan2(dy, dx)


def lon_lat_to_cartesian(lon, lat):
    """
    calculates x,y,z coordinates of a point on a sphere with
    radius R = EARTH_RADIUS_METERS
    """
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)

    x =  EARTH_RADIUS_METERS * np.cos(lat_r) * np.cos(lon_r)
    y = EARTH_RADIUS_METERS * np.cos(lat_r) * np.sin(lon_r)
    z = EARTH_RADIUS_METERS * np.sin(lat_r)
    return x,y,z



#nvectors.shape = (3, nx, ny)
def get_coefs_between(nvectors1, nvectors2):
    return np.array([1.0 / (get_angle_between_vectors(v1, v2) * EARTH_RADIUS_METERS ) ** 2.0 for v1, v2 in zip(nvectors1, nvectors2)])



def test():
    p1 = GeoPoint(-86.67,36.12)
    p2 = GeoPoint(-118.4, 33.94)
    print get_distance_in_meters(p1, p2)
    print get_distance_in_meters(p1.longitude, p1.latitude, p2.longitude, p2.latitude)
    print 'Theoretical distance: %f km' % 2887.26

if __name__ == "__main__":
    test()
    print "Hello World"
