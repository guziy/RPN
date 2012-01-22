__author__="huziy"
__date__ ="$13 juil. 2010 13:36:11$"

from copy import deepcopy
from math import radians
from math import *


class GeoPoint(object):
    '''
    Represents geographical point of (lon, lat)
    '''
    def __init__(self, longitude = 0.0, latitude = 0.0):
        self.longitude = float( longitude )
        self.latitude = float( latitude )

    def clone(self):
        return deepcopy(self)

    #http://en.wikipedia.org/wiki/N-vector#Converting_latitude.2Flongitude_to_n-vector
    def get_nvector(self):
        rad_lon = radians(self.longitude)
        rad_lat = radians(self.latitude)
        return [ cos(rad_lat) * cos(rad_lon), cos(rad_lat) * sin(rad_lon), sin(rad_lat) ]

    def __str__(self):
        return '(lon = %f , lat = %f )' % (self.longitude, self.latitude)

def test():
    p1 = GeoPoint(1,1)
    p2 = p1.clone()

    print p1 == p2
    print p1 is p2
    print p1, p1.clone()

    p2.longitude = 5
    print p1.longitude
    print p2.longitude 


if __name__ == "__main__":
    test()
