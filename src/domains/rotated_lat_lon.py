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

        self.mean_earth_radius_m_crcm5 = 0.637122e7 # mean earth radius used in the CRCM5 model for area calculation

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


    def write_coords_to_rpn(self, rpnObj, x, y):
        """
        x, y - 1d coordinates in rotated system
        """
	x = np.array(x)
        x.shape = (len(x),1)
        y = np.array(y)
        y.shape = (1, len(y)) 

        rpnObj.write_2D_field(name="^^", grid_type="E", data=y, typ_var="X", level = 0, ip = range(100,103),
            lon1=self.lon1, lat1 = self.lat1, lon2 = self.lon2, lat2 = self.lat2, label="")

        rpnObj.write_2D_field(name=">>", grid_type="E", data=x, typ_var="X", level = 0, ip = range(100, 103),
            lon1=self.lon1, lat1 = self.lat1, lon2 = self.lon2, lat2 = self.lat2, label = "")

        info = rpnObj.get_current_info()
        ip_xy = map(lambda x: x.value, info["ip"])
        ig = ip_xy + [0]
        return ig



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

    def get_areas_of_gridcells(self, dlon, dlat, nx, ny, latref, jref):

        """
        dlon, dlat, lonref, latref - are in degrees
        iref and jref are counted starting from 1, put here direct values grom gemclim_settings.nml
        """
        dx = np.radians(dlon)
        dy = np.radians(dlat)

        latref_rad = np.radians(latref)

        lats2d = np.zeros((nx, ny))
        for j in range(ny):
            lat = latref_rad + (j - jref + 1) * dy
            lats2d[:,j] = lat

        return self.mean_earth_radius_m_crcm5 ** 2 * np.cos(lats2d) * dx * dy





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
  
