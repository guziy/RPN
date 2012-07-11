__author__ = 'huziy'

import numpy as np

class GridParams:
    def __init__(self, lonr = 0, latr = 0, iref = 1, jref = 1,
                 nx = -1, ny = -1, dx = 0.1, dy = 0.1):
        self.lonr = lonr
        self.latr = latr
        self.iref = iref
        self.jref = jref
        self.nx = nx
        self.ny = ny

        self.dx = dx
        self.dy = dy

    def get_ll_point(self, marginx = 0, marginy = 0):
        return self.lonr + (marginx - self.iref) *  self.dx, self.latr + (marginy - self.jref) * self.dy

    def get_ur_point(self, marginx = 0, marginy = 0):
        return self.lonr + (self.nx - 1 - marginx - self.iref) *  self.dx, \
               self.latr + (self.ny - 1 - marginy - self.jref) * self.dy


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  