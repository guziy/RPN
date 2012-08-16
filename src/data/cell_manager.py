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

        for i in range(nx):
            for j in range(ny):
                i_next, j_next = direction_and_value.to_indices(i,j, flow_dirs[i][j])
                next_cell = None
                if 0 <= i_next < nx:
                    if 0 <= j_next < ny:
                        next_cell = self.cells[i_next][j_next]
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




def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  