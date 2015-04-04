__author__="huziy"
__date__ ="$25 mai 2010 18:31:59$"

from math import atan2
import os
from math import radians
from math import *

import application_properties
application_properties.set_current_directory()

EARTH_RADIUS = 6371.01

def get_distance_km(lon1, lat1, lon2, lat2):
    dlon = radians( lon1 - lon2 )
    rlat1 = radians(lat1)
    rlat2 = radians(lat2)

    y = ((cos(rlat1) * sin(dlon)) ** 2.0 + (cos(rlat1) * sin(rlat2) - sin(rlat1) * cos(rlat2) * cos(dlon)) ** 2.0) ** 0.5
    x = sin(rlat1) * sin(rlat2) + cos(rlat1) * cos(rlat2) * cos(dlon)

    return atan2(y, x) * EARTH_RADIUS


class LatLonHolder():
    def __init__(self, path_to_data_folder = 'data'):
        base = path_to_data_folder + os.sep + 'coordinates' + os.sep
        lon_path = base + 'longitudes.txt'
        lat_path = base + 'latitudes.txt'

        self.longitudes = self._read_lon_lat( lon_path, subtract_360 = True )
        self.latitudes = self._read_lon_lat( lat_path )
        self._get_nx_ny_from_file(lat_path)
        

    def get_num_cells_along_x(self):
        return self.Nx

    def get_num_cells_along_y(self):
        return self.Ny

    def get_ix(self, composite_index):
        """
        ix goes from 1 to Nx
        """
        return composite_index % self.Nx + 1

    def get_iy(self, composite_index):
        return composite_index / self.Nx + 1

    def get_lon_lat(self, ix, iy):
        composite_index = self._get_composite_index(ix, iy)
        return [self.longitudes[composite_index], self.latitudes[composite_index]]

    def _get_composite_index(self, ix, iy):
        return (ix - 1) + (iy - 1) * self.Nx

    #private methods
    def _read_lon_lat(self, path, subtract_360 = False):
        """
        read longitudes or latitudes of the domin cells,
        from the specified path
        """
        result = []
        f = open(path)
        f.readline()
        for line in f:
            if line.strip() == '':
                continue

            start = 1
            fields = line.split()
            if fields[0].endswith('='):
                start += 1

            for field in fields[start:]:
                lat_lon = float(field)
                if subtract_360:
                    lat_lon = lat_lon - 360
                result.append(lat_lon)
        return result


    def _get_nx_ny_from_file(self, path): 
        file = open(path)
        lines = file.readlines()
        ny = len(lines) - 1
        #account for the empty lines
        for line in lines:
            if line.strip() == '':
                ny -= 1

        line = lines[1]
        fields = line.split()
        if fields[0].endswith('='):
            nx = len(fields) - 2
        else:
            nx = len(fields) - 1

        self.Nx = nx
        self.Ny = ny




if __name__ == "__main__":
    print(get_distance_km(-86.67, 36.12, -118.40, 33.94))
    holder = LatLonHolder()
    print(holder.get_num_cells_along_x())
    print(holder.get_num_cells_along_y())