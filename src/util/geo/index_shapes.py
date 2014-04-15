__author__ = 'huziy'


class IndexPoint(object):
    def __init__(self, i = None, j = None):
        """
        :param i: horizontal index starting from 0
        :param j: vertical ---//---
        """
        self.i = int(i)
        self.j = int(j)


class IndexRectangle(object):
    def __init__(self, lower_left_point = None, width = 0, height = 0):
        """
        :param lower_left_point: IndexPoint representing the lower left corner of the rectangle
        :param width: width in index space
        :param height: height ---//---
        """
        self.lower_left_point = lower_left_point
        self.width = int(width)
        self.height = int(height)