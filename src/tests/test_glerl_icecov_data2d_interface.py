import nose
from nemo.glerl_icecov_data2d_interface import GLERLIceCoverManager

__author__ = 'huziy'

import numpy as np
from nose.tools import eq_, assert_almost_equal


class TestGLERLIceCoverManager(GLERLIceCoverManager):
    def __init__(self):
        super(TestGLERLIceCoverManager, self).__init__()


    def test_lon_lat_to_ij(self):
        """
        Test lon/lat to index conversion
        """
        self.ncols = 1024
        lat, lon = 50.6027, -75.8690
        i, j = self.lon_lat_to_ij(lon, lat)
        assert i == 1023, "ix index is wrong {} instead of {}".format(i, 1023)
        assert j == 1023, "jy index is wrong {} instead of {}".format(j, 1023)



    def test_lon_lat_to_ij_with_header_1024x1024(self):
        """
        Test if lon_lat_to_ij, conforms to the header info (1024x1024)
        """

        self.nrows = 1024
        self.ncols = 1024
        self.xllcorner = -10288021.9553
        self.yllcorner = 4675974.1583
        self.cellsize = 1800


        # assert np.radians()


    def test_if_inversible(self):
        """
        test if F^-1(F(lon, lat)) == (lon, lat)

        """
        lat, lon = 50.6027, -75.8690

        self.ncols = 1024

        i, j = self.lon_lat_to_ij(lon, lat)

        for c1, c2, cname in zip((lon, lat), self.ij_to_lon_lat(i, j), ("lon", "lat")):
            assert_almost_equal(c1, c2, msg="{} is not the same after F^-1(F(.))", places=4)


    def test_lon_lat_to_ij_with_header_516x510(self):
        """
        Test if lon_lat_to_ij, conforms to the header info (516x510)
        """

        self.nrows = 516
        self.ncols = 510
        self.xllcorner = -649446.25
        self.yllcorner = 3306260
        self.cellsize = 2550

        # Lower left corner
        lon = -92.36612
        lat = 38.84815

        i, j = self.lon_lat_to_ij(lon, lat)

        print((self.ij_to_lon_lat(0, 0)))

        # eq_(0, i)
        # eq_(0, j)
        msg = "The indices became equal somehow but the equations are not applicable to the 516x510 grid"
        nose.tools.assert_not_equal((0, 0), (i, j), msg=msg)

        lon = -75.69930
        lat = 50.53781

        i, j = self.lon_lat_to_ij(lon, lat)
        nose.tools.assert_not_equal((i, j), (self.nrows - 1, self.ncols - 1))


