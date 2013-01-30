from data.cell import Cell
from util import direction_and_value

__author__ = 'huziy'

import numpy as np

#responsible for creating conected cells, and for quering subregions
class CellManager:
    def __init__(self, nx, ny, flow_dirs):
        self.cells = []
        self.nx = nx
        self.ny = ny
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


        pass

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



def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  