__author__ = "huziy"
__date__ ="$16 juil. 2010 14:44:08$"


from util.geo.lat_lon import get_distance_in_meters
import numpy as np
import pylab


SECONDS_PER_DEGREE = 60.0 * 60.0
DEFAULT_SHAPE = (100, 100)

from util.geo.GeoPoint import GeoPoint

class LonLatGrid(object):
    '''
    represents longitude, latitude grid
    grid_shape = (nlongitudes, nlatitudes)
    '''

    def __init__(self, lower_left_point = GeoPoint(),
                       grid_shape = DEFAULT_SHAPE,
                       resolution_seconds = 300.0):
                           
        self._delta = resolution_seconds / SECONDS_PER_DEGREE
        self._lower_left = lower_left_point

        max_longitude = (grid_shape[0] - 1) * self._delta + lower_left_point.longitude
        max_latitude = (grid_shape[1] - 1) * self._delta + lower_left_point.latitude
        self._longitudes = np.linspace(lower_left_point.longitude, max_longitude, grid_shape[0])
        self._latitudes = np.linspace(lower_left_point.latitude, max_latitude, grid_shape[1])
 

    def get_1d_longitudes(self):
        return self._longitudes

    def get_1d_latitudes(self):
        return self._latitudes 

    def get_2d_longitudes(self):
        return pylab.meshgrid(self._latitudes, self._longitudes)[1]

    def get_2d_latitudes(self):
        return pylab.meshgrid(self._latitudes, self._longitudes)[0]

    def get_lons_lats_2d(self):
        lats, lons = pylab.meshgrid(self._latitudes, self._longitudes)
        return lons, lats 

    def interpolate_from_grid_to_point(self, data, longitude, latitude):
        lons = self._longitudes
        lats = self._latitudes
        the_delta = self._delta

        lon_min, lon_max = lons[0], lons[-1]
        lat_min, lat_max = lats[0], lats[-1]


        lon_indices = []
        if longitude > lon_min and longitude < lon_max:
            i1 = int( (longitude - lon_min) / the_delta )
            lon_indices.append( i1 )
            lon_indices.append( i1 + 1 )
        elif longitude >= lon_max:
            lon_indices.append(-1) #last element in the row
        elif longitude <= lon_min:
            lon_indices.append(0)


        lat_indices = []
        if latitude > lat_min and latitude < lat_max:
            i1 = int( (latitude - lat_min) / the_delta )
            lat_indices.append( i1 )
            lat_indices.append( i1 + 1 )
        elif latitude >= lat_max:
            lat_indices.append(-1) #last element in the row
        elif latitude <= lat_min:
            lat_indices.append(0)

        result = 0.0
        the_norm = 0.0
        for i in lon_indices:
            for j in lat_indices:
                the_lon = lon_min + i * the_delta if i >= 0 else lon_max
                the_lat = lat_min + j * the_delta if j >= 0 else lat_max

                d = get_distance_in_meters(the_lon, the_lat, longitude, latitude)
                if d == 0:
                    return data[i,j]

                coef = 1.0 / d ** 2.0
                result += coef * data[i,j]
                the_norm += coef
        return result / the_norm




    def _get_longitude_difference(self, lon1, lon2):
        result = np.abs(lon1 - lon2)
        result = min(result, 360 - result)
        return result

    def _get_latitude_difference(self, lat1, lat2):
        result = np.abs(lat1 - lat2)
        result = min(result, 180 - result)
        assert result >= 0
        return result



    def get_indices_of_the_mesh_point(self, mesh_point = None, longitude = 0, latitude = 0):
        if mesh_point == None:
            dlon = longitude - self._lower_left.longitude
            dlat = latitude - self._lower_left.latitude 
        else:
            dlon = mesh_point.longitude - self._lower_left.longitude
            dlat = mesh_point.latitude - self._lower_left.latitude
        return int(dlon / self._delta) , int(dlat / self._delta)


    def __str__(self):
        nlons = len(self._longitudes)
        nlats = len(self._latitudes)

        max_lon = max(self._longitudes)
        max_lat = max(self._latitudes) 
        result = 'Grid(lon-lat): \n lower left point %s, \n (nlons = %d, nlats = %d) \n' % (self._lower_left, nlons, nlats )
        result += 'max longitude: %f, max latitude: %f' % (max_lon, max_lat) 
        return result


def test():
    grid = LonLatGrid(grid_shape = (100,150), lower_left_point = GeoPoint(longitude = -180, latitude = -90))
    print(grid.get_2d_latitudes().shape)
    print(np.min(grid.get_2d_longitudes()))
    print(len(grid.get_1d_latitudes()) == 150)
    print(grid)


if __name__ == "__main__":
    test()