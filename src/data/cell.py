__author__ = 'huziy'

import numpy as np


class Cell:
    def __init__(self, i = -1, j = -1, flow_dir_value = -1):
        self.previous = []
        self.next = None
        self.i = i
        self.j = j
        self.flow_dir_value = flow_dir_value
        pass

    def set_next(self, next_cell):
        """
        :type next_cell: Cell
        """
        self.next = next_cell
        if next_cell is not None:
            assert self not in next_cell.previous
            next_cell.previous.append(self)

    def get_upstream_cells(self):
        """
        get all upstream cells of the current cell
        including itself
        """
        res = [self]
        for p in self.previous:
            res.extend(p.get_upstream_cells())
            res.append(p)
        return res

    def get_ij(self):
        return self.i, self.j

    def is_downstream_for(self, aCell):
        """
        :type aCell: Cell
        """
        current = aCell
        while current is not None:
            current = current.next
            if current == self:
                return True
        return False
        pass

    def is_upstream_for(self, aCell):
        pass

def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  