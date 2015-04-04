__author__ = 'huziy'

import numpy as np


class FormatCoords():
    def __init__(self, the_field):
        """
        Formats coordinates of the mouse pointer in order to follow the value at
        the mouse position
        """
        self.data = the_field
        self.nrows, self.ncols = the_field.shape

    def __call__(self, x, y):
        return self._format_coord(x, y)


    def _format_coord(self, x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if 0 <= col < self.ncols and 0 <= row < self.nrows:
            z = self.data[row,col]
            return 'x=%1.4f, y=%1.4f, z=%g'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)

def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print("Hello world")
  