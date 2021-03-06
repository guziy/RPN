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
    DEFAULT_DRAINAGE_AREA_RELDIFF_MIN = 0.1


    # responsible for creating conected cells, and for quering subregions
    def __init__(self, flow_dirs, nx=None, ny=None,
                 lons2d=None,
                 lats2d=None,
                 accumulation_area_km2=None):
        self.cells = []
        self.lons2d = lons2d
        self.lats2d = lats2d
        self.flow_directions = flow_dirs

        self.accumulation_area_km2 = accumulation_area_km2

        # calculate characteristic distance
        if not any([None is arr for arr in [self.lats2d, self.lons2d]]):
            v1 = lat_lon.lon_lat_to_cartesian(self.lons2d[0, 0], self.lats2d[0, 0])
            v2 = lat_lon.lon_lat_to_cartesian(self.lons2d[1, 1], self.lats2d[1, 1])
            dv = np.array(v2) - np.array(v1)
            self.characteristic_distance = np.sqrt(np.dot(dv, dv))

            x, y, z = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
            self.kdtree = cKDTree(list(zip(x, y, z)))

        if None not in [nx, ny]:
            self.nx = nx
            self.ny = ny
        else:
            nx, ny = flow_dirs.shape
            self.nx, self.ny = flow_dirs.shape

        for i in range(nx):
            self.cells.append(list([Cell(i=i, j=j, flow_dir_value=flow_dirs[i, j]) for j in range(ny)]))

        self._without_next_mask = np.zeros((nx, ny), dtype=np.int)
        self._wo_next_wo_prev_mask = np.zeros((nx, ny), dtype=np.int)  # mask of the potential outlets
        for i in range(nx):
            if i % 100 == 0:
                print("Created {}/{}".format(i, nx))
            for j in range(ny):
                i_next, j_next = direction_and_value.to_indices(i, j, flow_dirs[i][j])
                next_cell = None
                if 0 <= i_next < nx:
                    if 0 <= j_next < ny:
                        next_cell = self.cells[i_next][j_next]

                self._without_next_mask[i, j] = int(next_cell is None)
                self.cells[i][j].set_next(next_cell)


    def get_outlet_mask_array(self, lower_accumulation_index_limit=5):
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


    def get_model_points_of_outlets(self, lower_accumulation_index_limit=5):
        """
        Does the same thing as self.get_oulet_mask_array, except the result is a list of ModelPoint objects
        :param lower_accumulation_index_limit:
        """
        omask = self.get_outlet_mask_array(lower_accumulation_index_limit=lower_accumulation_index_limit)
        return [ModelPoint(ix=i, jy=j, longitude=self.lons2d[i, j], latitude=self.lats2d[i, j])
                for i, j in zip(*np.where(omask))]


    def get_accumulation_index(self):
        # returns a field of the number of cells flowing into a given cell
        # (based on the list of cells representing current domain)
        result = np.zeros((self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                the_cell = self.cells[i][j]
                assert isinstance(the_cell, Cell)
                if the_cell.__next__ is not None:
                    continue

                result[i, j] = the_cell.get_number_of_upstream_cells()
                for acell in the_cell.get_upstream_cells():
                    result[acell.i, acell.j] = acell.get_number_of_upstream_cells()

        return result


    def get_mask_of_upstream_cells_connected_with_by_indices(self, ix, jy):
        """
        Note: only upstream cells are checked
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


    def get_outlet_mask(self, rout_domain_mask=None):
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


            # do not consider outlets, since they contain ocean points, the runoff for which can be
            # negative, and this is not accounted for in the routing scheme
            rout_mask[io, jo] = 0

        return rout_mask


    def get_lake_model_points_for_stations(self, station_list, lake_fraction=None,
                                           nneighbours=8):

        """
        For lake levels we have a bit different search algorithm since accumulation area is not a very sensible param to compare
        :return {station: list of corresponding model points}

        :param station_list:
        :param lake_fraction:
        :param drainaige_area_reldiff_limit:
        :param nneighbours:
        :return: :raise Exception:
        """

        station_to_model_point_list = {}
        nx, ny = self.lons2d.shape
        i1d, j1d = list(range(nx)), list(range(ny))
        j2d, i2d = np.meshgrid(j1d, i1d)
        i_flat, j_flat = i2d.flatten(), j2d.flatten()

        for s in station_list:
            mp_list = []

            assert isinstance(s, Station)
            x, y, z = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)
            dists, inds = self.kdtree.query((x, y, z), k=nneighbours)
            if nneighbours == 1:
                dists = [dists]
                inds = [inds]

            for d, i in zip(dists, inds):
                ix = i_flat[i]
                jy = j_flat[i]
                mp = ModelPoint(ix=ix, jy=jy)

                mp.longitude = self.lons2d[ix, jy]
                mp.latitude = self.lats2d[ix, jy]

                mp.distance_to_station = d
                if lake_fraction is not None:
                    if lake_fraction[ix, jy] <= 0.001:  # skip the model point if almost no lakes inisde
                        continue

                    mp.lake_fraction = lake_fraction[ix, jy]
                mp_list.append(mp)

            if lake_fraction is not None:
                lf = 0.0
                for mp in mp_list:
                    lf += mp.lake_fraction

                if lf <= 0.001:
                    continue

            station_to_model_point_list[s] = mp_list
            print("Found model point for the station {0}".format(s))

        return station_to_model_point_list


    def get_model_points_for_stations(self, station_list, lake_fraction=None,
                                      drainaige_area_reldiff_limit=None, nneighbours=4):
        """
        returns a map {station => modelpoint} for comparison modeled streamflows with observed
        :rtype   dict
        """


        # if drainaige_area_reldiff_limit is None:
        #     drainaige_area_reldiff_limit = self.DEFAULT_DRAINAGE_AREA_RELDIFF_MIN

        # if nneighbours == 1:
        #     raise Exception("Searching over 1 neighbor is not very secure and not implemented yet")

        station_to_model_point = {}
        model_acc_area = self.accumulation_area_km2
        model_acc_area_1d = model_acc_area.flatten()

        grid = np.indices(model_acc_area.shape)

        for s in station_list:

            x, y, z = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)

            if s.drainage_km2 is None or nneighbours == 1:
                # return the closest grid point

                dists, inds = self.kdtree.query((x, y, z), k=1)
                ix, jy = [g1.flatten()[inds] for g1 in grid]

                imin = 0
                dists = [dists]

                if s.drainage_km2 is None:
                    print("Using the closest grid point, since the station does not report its drainage area: {}".format(s))

            else:

                if s.drainage_km2 < self.characteristic_distance ** 2 * 1e-12:
                    print("skipping {0}, because drainage area is too small: {1} km**2".format(s.id, s.drainage_km2))
                    continue

                assert isinstance(s, Station)
                dists, inds = self.kdtree.query((x, y, z), k=nneighbours)

                deltaDaMin = np.min(np.abs(model_acc_area_1d[inds] - s.drainage_km2))

                # this returns a  list of numpy arrays
                imin = np.where(np.abs(model_acc_area_1d[inds] - s.drainage_km2) == deltaDaMin)[0][0]

                # deltaDa2D = np.abs(self.accumulation_area_km2 - s.drainage_km2)

                # ij = np.where(deltaDa2D == deltaDaMin)
                ix, jy = grid[0].flatten()[inds][imin], grid[1].flatten()[inds][imin]

                # check if it is not global lake cell (move downstream if it is)
                if lake_fraction is not None:
                    while lake_fraction[ix, jy] >= infovar.GLOBAL_LAKE_FRACTION:
                        di, dj = direction_and_value.flowdir_values_to_shift(self.flow_directions[ix, jy])
                        ix, jy = ix + di, jy + dj


                # check if the gridcell is not too far from the station
                # if dists[imin] > 2 * self.characteristic_distance:
                #    continue

                # check if difference in drainage areas is not too big less than 10 %
                if drainaige_area_reldiff_limit is not None and deltaDaMin / s.drainage_km2 > drainaige_area_reldiff_limit:
                    print("Drainage area relative difference is too high, skipping {}.".format(s.id))
                    print(deltaDaMin / s.drainage_km2, deltaDaMin, s.drainage_km2)
                    continue



            mp = ModelPoint()
            mp.ix = ix
            mp.jy = jy

            mp.longitude = self.lons2d[ix, jy]
            mp.latitude = self.lats2d[ix, jy]

            mp.accumulation_area = self.accumulation_area_km2[ix, jy]

            try:
                mp.distance_to_station = dists[imin]
            except TypeError:
                mp.distance_to_station = float(dists)

            station_to_model_point[s] = mp

            print("mp.accumulation_area_km2={}; s.drainage_km2={}".format(mp.accumulation_area, s.drainage_km2))

            print("Found model point for the station {0}".format(s))

        return station_to_model_point

    def get_upstream_polygons_for_points(self, model_point_list,
                                         xx: np.ndarray=None, yy: np.ndarray=None):

        """

        : return : list of 2d arrays representing coordinates of the vertices of the edges
        :param model_point_list:
        :type model_point_list: iterable[ModelPoint]
        :rtype : list
        """

        res = []


        for mp in model_point_list:


            mask = self.get_mask_of_upstream_cells_connected_with_by_indices(mp.ix, mp.jy)

            edges = []  # [[(x1, x2), (y1, y2)], ...]
            for i0, j0 in zip(*np.where(mask == 1)):
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue

                        # Skip diagonal neighbours
                        if abs(di * dj) == 1:
                            continue

                        i = i0 + di
                        j = j0 + dj

                        if mask[i, j] != 1:

                            if j == j0:
                                x1 = x2 = (xx[i, j] + xx[i0, j]) * 0.5
                                y1 = (yy[i0, j0] + yy[i0, j0 - 1] + yy[i, j] + yy[i, j - 1]) * 0.25
                                y2 = (yy[i0, j0] + yy[i0, j0 + 1] + yy[i, j] + yy[i, j + 1]) * 0.25
                            else:  # i == i0
                                y1 = y2 = (yy[i, j0] + yy[i, j]) * 0.5
                                x1 = (xx[i0, j0] + xx[i0 - 1, j0] + xx[i, j] + xx[i - 1, j]) * 0.25
                                x2 = (xx[i0, j0] + xx[i0 + 1, j0] + xx[i, j] + xx[i + 1, j]) * 0.25

                            edges.append([(x1, x2), (y1, y2)])
            res.append(edges)

        # List of lists of edges
        return res




def main():
    # TODO: implement
    pass


if __name__ == "__main__":
    main()
    print("Hello world")
  