from datetime import datetime, timedelta
from netCDF4 import Dataset, MFDataset
import os
import itertools

__author__ = 'huziy'


class AnuSplinManager:

    def __init__(self, folder_path = ""):
        self.lons2d = None
        self.lats2d = None
        self.folder_path = folder_path
        self.fname_format = "ANUSPLIN_latlon_pcp_%Y_%m.nc"
        pass


    def _read_lon_lats(self):
        """
        Read the lons and lats, 2d from the first file
        """
        the_date = datetime(1980, 1, 1)
        fPath = os.path.join(self.folder_path, the_date.strftime(self.fname_format))
        ds = Dataset(fPath)
        self.lons2d = ds.variables["lon"][:].transpose()
        self.lats2d = ds.variables["lat"][:].transpose()
        ds.close()

    def _get_year(self, fName):
        return datetime.strptime(fName, self.fname_format)

    def get_daily_means_array(self, start_year = 1980, end_year = 2010):
        """
        :rtype : np.array, list of datetime objects
        :param start_year: start year of the averaging period
        :param end_year: end year of the averaging period
        :return the array of daily means and the array of times corresponding to those daily means
        """

        stamp_year = 2001
        one_day = timedelta(days = 1)
        the_date = datetime(stamp_year, 1, 1)

        for month in range(1, 13):
            pass



        # TODO: implement

        pass