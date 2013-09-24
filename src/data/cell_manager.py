from scipy.spatial.ckdtree import cKDTree
from crcm5 import infovar
from crcm5.model_point import ModelPoint
from data.cehq_station import Station
from data.cell import Cell
from util import direction_and_value
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np

#responsible for creating conected cells, and for quering subregions
class CellManager:

    def __init__(self, flow_dirs, nx = None, ny = None,
                 lons2d = None,
                 lats2d = None,
                 accumulation_area_km2 = None
                 ):
        self.cells = []
        self.lons2d = lons2d
        self.lats2d = lats2d

        self.accumulation_area_km2 = accumulation_area_km2

        #calculate characteristic distance
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
            nx, ny  = flow_dirs.shape
            self.nx, self.ny = flow_dirs.shape

        for i in range(nx):
            self.cells.append(list([Cell(i = i, j = j, flow_dir_value = flow_dirs[i,j]) for j in range(ny)]))

        self._without_next_mask = np.zeros((nx, ny), dtype=np.int)
        self._wo_next_wo_prev_mask = np.zeros((nx, ny), dtype=np.int) #mask of the potential outlets
        for i in range(nx):
            for j in range(ny):
                i_next, j_next = direction_and_value.to_indices(i,j, flow_dirs[i][j])
                next_cell = None
                if 0 <= i_next < nx:
                    if 0 <= j_next < ny:
                        next_cell = self.cells[i_next][j_next]


                self._without_next_mask[i,j] = int(next_cell is None)
                self.cells[i][j].set_next(next_cell)



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



    def get_mask_of_cells_connected_with(self, aCell):
        """
        returns 2d array indicating 1 where there is a cell connected to aCell and 0 where it is not
        :type aCell: Cell
        """
        assert isinstance(aCell, Cell)
        all_upstream = aCell.get_upstream_cells()
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


        pass

    def get_coherent_rout_domain_mask(self, outlet_mask):
        """
        ignore the cells which only have 1 previous cell (i.e. itself)
        """
        i,j = np.where(outlet_mask == 1)
        rout_mask = np.zeros(outlet_mask.shape).astype(int)
        for io, jo in zip(i,j):
            upstream = self.cells[io][jo].get_upstream_cells()
            if len(upstream) <= 1:
                continue

            for the_cell in upstream:
                rout_mask[the_cell.i, the_cell.j] = 1


            #do not consider outlets, since they contain ocean points, the runoff for which can be
            #negative, and this is not accounted for in the routing scheme
            rout_mask[io,jo] = 0

        return rout_mask

    def get_model_points_for_stations(self, station_list, lake_fraction = None):
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
            dists, inds = self.kdtree.query((x, y, z), k = 8)


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
            if deltaDaMin / s.drainage_km2 > 0.1:
                continue

            mp = ModelPoint()
            mp.ix = ix
            mp.jy = jy

            mp.longitude = self.lons2d[ix, jy]
            mp.latitude = self.lats2d[ix, jy]

            mp.accumulation_area = self.accumulation_area_km2[ix, jy]
            mp.distance_to_station = dists[imin]

            station_to_model_point[s] = mp

        return station_to_model_point


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  