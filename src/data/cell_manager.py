from scipy.spatial.ckdtree import cKDTree
from crcm5 import infovar
from crcm5.model_point import ModelPoint
from data.cehq_station import Station
from data.cell import Cell
from util import direction_and_value
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np


class CellManager:
    #responsible for creating conected cells, and for quering subregions
    def __init__(self, flow_dirs, nx = None, ny = None,
                 lons2d = None,
                 lats2d = None,
                 accumulation_area_km2 = None
                 ):
        self.cells = []
        self.lons2d = lons2d
        self.lats2d = lats2d
        self.flow_directions = flow_dirs

        self.accumulation_area_km2 = accumulation_area_km2

        #calculate characteristic distance
        if None not in [self.lats2d, self.lons2d]:
            v1 = lat_lon.lon_lat_to_cartesian(self.lons2d[0, 0], self.lats2d[0, 0])
            v2 = lat_lon.lon_lat_to_cartesian(self.lons2d[1, 1], self.lats2d[1, 1])
            dv = np.array(v2) - np.array(v1)
            self.characteristic_distance = np.sqrt(np.dot(dv, dv))

            x, y, z = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
            self.kdtree = cKDTree(zip(x, y, z))

        if not None in [nx, ny]:
            self.nx = nx
            self.ny = ny
        else:
            nx, ny = flow_dirs.shape
            self.nx, self.ny = flow_dirs.shape

        for i in range(nx):
            self.cells.append(list([Cell(i = i, j = j, flow_dir_value = flow_dirs[i, j]) for j in range(ny)]))

        self._without_next_mask = np.zeros((nx, ny), dtype=np.int)
        self._wo_next_wo_prev_mask = np.zeros((nx, ny), dtype=np.int)  # mask of the potential outlets
        for i in range(nx):
            for j in range(ny):
                i_next, j_next = direction_and_value.to_indices(i, j, flow_dirs[i][j])
                next_cell = None
                if 0 <= i_next < nx:
                    if 0 <= j_next < ny:
                        next_cell = self.cells[i_next][j_next]


                self._without_next_mask[i, j] = int(next_cell is None)
                self.cells[i][j].set_next(next_cell)



    def get_outlet_mask_array(self, lower_accumulation_index_limit = 5):
        """
        returns a mask 2d bool array which is True where accumulation index is greater then (>)
        lower_accumulation_index_limit
        """

        is_outlet_candidate = (self.flow_directions <= 0) | (self.flow_directions >= 129)

        not_valid_dir_value = (self.flow_directions != 1)
        for i in range(1, 8):
            not_valid_dir_value &= self.flow_directions != 2 ** i

        is_outlet_candidate |= not_valid_dir_value

        return (self.get_accumulation_index() > lower_accumulation_index_limit) & is_outlet_candidate


    def get_model_points_of_outlets(self, lower_accumulation_index_limit = 5):
        """
        Does the same thing as self.get_oulet_mask_array, except the result is a list of ModelPoint objects
        :param lower_accumulation_index_limit:
        """
        omask = self.get_outlet_mask_array(lower_accumulation_index_limit = lower_accumulation_index_limit)
        return [ModelPoint(ix=i, jy=j, longitude=self.lons2d[i, j], latitude=self.lats2d[i, j])
                for i, j in zip(*np.where(omask))]


    def get_accumulation_index(self):
        #returns a field of the number of cells flowing into a given cell
        #(based on the list of cells representing current domain)
        result = np.zeros((self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                the_cell = self.cells[i][j]
                assert isinstance(the_cell, Cell)
                if the_cell.next is not None:
                    continue

                result[i, j] = the_cell.get_number_of_upstream_cells()
                for acell in the_cell.get_upstream_cells():
                    result[acell.i, acell.j] = acell.get_number_of_upstream_cells()

        return result





    def get_mask_of_cells_connected_with_by_indices(self, ix, jy):
        """
        returns 2d array indicating 1 where there is a cell connected to aCell and 0 where it is not
        ix, jy - horizontal and vertical indices of the Cell to which the upstream are sought
        """
        all_upstream = self.cells[ix][jy].get_upstream_cells()
        the_mask = np.zeros((self.nx, self.ny)).astype(int)
        for uc in all_upstream:
            the_mask[uc.i, uc.j] = 1
        return the_mask



    def get_mask_of_cells_connected_with(self, acell):
        """
        returns 2d array indicating 1 where there is a cell connected to aCell and 0 where it is not
        :type acell: Cell
        """
        assert isinstance(acell, Cell)
        all_upstream = acell.get_upstream_cells()
        the_mask = np.zeros((self.nx, self.ny)).astype(int)
        for uc in all_upstream:
            the_mask[uc.i, uc.j] = 1
        return the_mask



    def get_outlet_mask(self, rout_domain_mask = None):
        """
        returns a matrix of 0/1, 1 - where you have outlets
        note, that there is no way to distinguish ocean cells and outlets at this stage, if the
        rout_domain_mask
        """

        if rout_domain_mask is not None:
            return self._without_next_mask * rout_domain_mask
        else:
            return self._without_next_mask



    def get_coherent_rout_domain_mask(self, outlet_mask):
        """
        ignore the cells which only have 1 previous cell (i.e. itself)
        """
        i, j = np.where(outlet_mask == 1)
        rout_mask = np.zeros(outlet_mask.shape).astype(int)
        for io, jo in zip(i, j):
            upstream = self.cells[io][jo].get_upstream_cells()
            if len(upstream) <= 1:
                continue

            for the_cell in upstream:
                rout_mask[the_cell.i, the_cell.j] = 1


            #do not consider outlets, since they contain ocean points, the runoff for which can be
            #negative, and this is not accounted for in the routing scheme
            rout_mask[io, jo] = 0

        return rout_mask

    def get_model_points_for_stations(self, station_list, lake_fraction = None, drainaige_area_reldiff_limit = 0.1):
        """
        returns a map {station => modelpoint} for comparison modeled streamflows with observed
        :rtype   dict
        """

        station_to_model_point = {}
        model_acc_area = self.accumulation_area_km2
        model_acc_area_1d = model_acc_area.flatten()


        for s in station_list:
            assert isinstance(s, Station)
            x, y, z = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)
            dists, inds = self.kdtree.query((x, y, z), k = 16)


            deltaDaMin = np.min(np.abs(model_acc_area_1d[inds] - s.drainage_km2))

            #this returns a  list of numpy arrays
            imin = np.where(np.abs(model_acc_area_1d[inds] - s.drainage_km2) == deltaDaMin)[0][0]

            deltaDa2D = np.abs(self.accumulation_area_km2 - s.drainage_km2)

            ij = np.where(deltaDa2D == deltaDaMin)

            ix, jy = ij[0][0], ij[1][0]

            #check if it is not global lake cell
            if lake_fraction is not None and lake_fraction[ix, jy] >= infovar.GLOBAL_LAKE_FRACTION:
                continue

            #check if the gridcell is not too far from the station
            if dists[imin] > 2 * self.characteristic_distance:
                continue

            #check if difference in drainage areas is not too big less than 10 %
            if deltaDaMin / s.drainage_km2 > drainaige_area_reldiff_limit:
                print deltaDaMin / s.drainage_km2, deltaDaMin, s.drainage_km2
                continue

            mp = ModelPoint()
            mp.ix = ix
            mp.jy = jy

            mp.longitude = self.lons2d[ix, jy]
            mp.latitude = self.lats2d[ix, jy]

            mp.accumulation_area = self.accumulation_area_km2[ix, jy]
            mp.distance_to_station = dists[imin]

            station_to_model_point[s] = mp
            print u"Found model point for the station {0}".format(s)

        return station_to_model_point


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  