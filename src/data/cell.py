__author__ = 'huziy'

import numpy as np


class Cell:
    def __init__(self, i = -1, j = -1):
        self.previous = []
        self.next = None
        self.i = i
        self.j = j
        pass

    def set_next(self, next_cell):
        """
        :type next_cell: Cell
        """
        self.next = next_cell
        if next_cell is not None:
            assert self not in next_cell.previous
            next_cell.previous.append(self)

def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  