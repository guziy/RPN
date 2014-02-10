#
__author__ = 'huziy'

import numpy as np
from util.geo import lat_lon


class RotatedLatLon():
    def __init__(self, lon1=180, lat1=0, lon2=180, lat2=0):
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

        self.mean_earth_radius_m_crcm5 = 0.637122e7  # mean earth radius used in the CRCM5 model for area calculation

        p1 = lat_lon.lon_lat_to_cartesian(lon1, lat1, R=1.0)
        p2 = lat_lon.lon_lat_to_cartesian(lon2, lat2, R=1.0)

        p1 = np.array(p1)
        p2 = np.array(p2)

        cross_prod = np.cross(p1, p2)
        dot_prod = np.dot(p1, p2)

        row0 = -np.array(p1) / np.sqrt(np.dot(p1, p1))
        e2 = (dot_prod * p1 - p2)
        row1 = e2 / np.sqrt(np.dot(e2, e2))
        row2 = cross_prod / np.sqrt(np.dot(cross_prod, cross_prod))
        self.rot_matrix = np.matrix([row0, row1, row2])
        assert isinstance(self.rot_matrix, np.matrix)


    def write_coords_to_rpn(self, rpnObj, x, y):
        """
        x, y - 1d coordinates in rotated system
        """
        x = np.array(x)
        x.shape = (len(x), 1)
        y = np.array(y)
        y.shape = (1, len(y))

        rpnObj.write_2D_field(name="^^", grid_type="E", data=y, typ_var="X", level=0, ip=range(100, 103),
                              lon1=self.lon1, lat1=self.lat1, lon2=self.lon2, lat2=self.lat2, label="")

        rpnObj.write_2D_field(name=">>", grid_type="E", data=x, typ_var="X", level=0, ip=range(100, 103),
                              lon1=self.lon1, lat1=self.lat1, lon2=self.lon2, lat2=self.lat2, label="")

        info = rpnObj.get_current_info
        ip_xy = map(lambda x: x.value, info["ip"])
        ig = ip_xy + [0]
        return ig


    def toProjectionXY(self, lon, lat):
        """
        Convert geographic lon/lat coordinates to the rotated lat lon coordinates
        """

        p = lat_lon.lon_lat_to_cartesian(lon, lat, R=1)
        p = self.rot_matrix * np.mat(p).T
        return lat_lon.cartesian_to_lon_lat(p.A1)


    def toGeographicLonLat(self, x, y):
        """
        convert geographic lat / lon to rotated coordinates
        """
        p = lat_lon.lon_lat_to_cartesian(x, y, R=1)
        p = self.rot_matrix.T * np.mat(p).T
        return lat_lon.cartesian_to_lon_lat(p.A1)


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
            lats2d[:, j] = lat

        return self.mean_earth_radius_m_crcm5 ** 2 * np.cos(lats2d) * dx * dy


    def get_south_pol_coords(self):
        rot_pole = self.rot_matrix * np.mat([0, 0, -1]).T
        return lat_lon.cartesian_to_lon_lat(rot_pole.A1)
        pass


    def get_north_pole_coords(self):
        """
        get true coordinates of the rotated north pole
        """
        rot_pole = self.rot_matrix * np.mat([0, 0, 1]).T
        return lat_lon.cartesian_to_lon_lat(rot_pole.A1)


    def get_true_pole_coords_in_rotated_system(self):
        """
        needed for lon_0 in basemap
        """
        rot_pole = self.rot_matrix.T * np.mat([0, 0, 1]).T
        return lat_lon.cartesian_to_lon_lat(rot_pole.A1)


def south_pole_coords_test():
    import application_properties

    application_properties.set_current_directory()
    rot_lat_lon = RotatedLatLon(lon1=-68, lat1=52, lon2=16.65, lat2=0.0)
    splon, splat = rot_lat_lon.get_south_pol_coords()

    #get some data
    from crcm5.model_data import Crcm5ModelDataManager

    base_data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff"
    base_data_manager = Crcm5ModelDataManager(samples_folder_path=base_data_path,
                                              all_files_in_samples_folder=True, need_cell_manager=True)

    lons2d = base_data_manager.lons2D
    lats2d = base_data_manager.lats2D

    acc = base_data_manager.accumulation_area_km2

    import Ngl

    #
    #  Open a workstation.
    #
    wks_type = "ps"
    wks = Ngl.open_wks(wks_type, "ngl05p")

    resources = Ngl.Resources()

    resources.tiXAxisString = "~F25~longitude"
    resources.tiYAxisString = "~F25~latitude"

    resources.cnFillOn = True     # Turn on contour fill.
    resources.cnLineLabelsOn = False    # Turn off line labels.
    resources.cnInfoLabelOn = False    # Turn off info label.

    resources.nglSpreadColorEnd = -2       # Don't include gray in contours.

    #resources.sfXCStartV = float(min(pf_lon))   # Define where contour plot
    #resources.sfXCEndV   = float(max(pf_lon))   # should lie on the map plot.
    #resources.sfYCStartV = float(min(pf_lat))
    #resources.sfYCEndV   = float(max(pf_lat))

    #resources.mpProjection = "LambertEqualArea"  # Change the map projection.
    #resources.mpCenterLonF = (pf_lon[pf_nlon-1] + pf_lon[0])/2
    #resources.mpCenterLatF = (pf_lat[pf_nlat-1] + pf_lat[0])/2

    resources.pmTickMarkDisplayMode = "Never"  # Turn off map tickmarks.

    resources.tiMainString = "~F26~January 1996 storm" # Set a title.

    resources.vpXF = 0.1    # Change the size and location of the
    resources.vpYF = 0.9    # plot on the viewport.
    resources.vpWidthF = 0.7
    resources.vpHeightF = 0.7

    resources.nglFrame = False # Don't advance frame.

    resources.mpCenterLonF = splon
    resources.mpCenterLatF = splat

    resources.tfDoNDCOverlay = True
    #resources.sfXArray          = lons2d
    #resources.sfYArray          = lats2d

    #TODO: it seems somewhat complicated to define the plot coordinates...
    map = Ngl.contour_map(wks, acc, resources)


def main():
    rll = RotatedLatLon(lon1=-68, lat1=52, lon2=16.65, lat2=0.0)
    print rll.rot_matrix

    prj = rll.toProjectionXY(0, 0)
    print prj
    print rll.toGeographicLonLat(prj[0], prj[1])

    print south_pole_coords_test()

    #TODO: implement
    pass


if __name__ == "__main__":
    main()
    print "Hello world"

