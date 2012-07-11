__author__ = 'huziy'

import numpy as np
from util.geo import lat_lon


class RotatedLatLon():


    def __init__(self, lon1 = 180, lat1 = 0, lon2 = 180, lat2 = 0):
        """
        Basis vactors of the rotated coordinate system in the original coord system
        e1 = -p1/|p1|                   =>   row0
        e2 = -( p2 - (p1, p2) * p1) / |p2 - (p1, p2) * p1| #perpendicular to e1, and lies in
        the plane parallel to the plane (p1^p2)  => row1
        e3 = [p1,p2] / |[p1, p2]| , perpendicular to the plane (p1^p2)          => row2
        """

        self.lon1 = lon1
        self.lon2 = lon2
        self.lat1 = lat1
        self.lat2 = lat2

        p1 = lat_lon.lon_lat_to_cartesian(lon1, lat1, R = 1.0)
        p2 = lat_lon.lon_lat_to_cartesian(lon2, lat2, R = 1.0)

        p1 = np.array(p1)
        p2 = np.array(p2)

        cross_prod = np.cross(p1, p2)
        dot_prod = np.dot(p1, p2)

        row0 = -np.array(p1) / np.sqrt(np.dot( p1, p1 ))
        e2 = (dot_prod * p1 - p2)
        row1 = e2 / np.sqrt(np.dot( e2, e2 ))
        row2 = cross_prod / np.sqrt( np.dot(cross_prod, cross_prod) )
        self.rot_matrix = np.matrix( [row0, row1, row2] )
        assert isinstance(self.rot_matrix, np.matrix)
        pass


    def toProjectionXY(self, lon, lat):
        """
        Convert geographic lon/lat coordinates to the rotated lat lon coordinates
        """

        p = lat_lon.lon_lat_to_cartesian(lon, lat, R = 1)
        p = self.rot_matrix * np.mat(p).T
        return lat_lon.cartesian_to_lon_lat(p.A1)


    def toGeographicLonLat(self, x, y):
        """
        convert geographic lat / lon to rotated coordinates
        """
        p = lat_lon.lon_lat_to_cartesian(x, y, R = 1)
        p = self.rot_matrix.T * np.mat( p ).T

        return lat_lon.cartesian_to_lon_lat(p.A1)

        pass





def main():

    rll = RotatedLatLon(lon1=-68, lat1=52, lon2=16.65, lat2=0.0)
    print rll.rot_matrix

    prj = rll.toProjectionXY(0,0)
    print prj
    print rll.toGeographicLonLat(prj[0], prj[1])
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  