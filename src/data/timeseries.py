from datetime import date, timedelta

__author__ = 'huziy'

import numpy as np

class TimeSeries:
    def __init__(self, data = None, time = None):

        if data is not None and time is not None:
            assert len(data) == len(time)
        else:
            assert [time, data] == [None, None]
        self.data = data
        self.time = time
        self.metadata = {}
        pass

    def get_mean(self, months = xrange(1,13)):
        """
        returns mean over the months speciifed in the
        months parameter
        """
        bool_vector = map(lambda x: x.month in months, self.time)
        indices = np.where(bool_vector)[0]
        return np.mean(self.data[indices])

    def get_monthly_normals(self):
        """
        returns the list of 12 monthly normals corresponding
        to the 12 months [0->Jan, ..., 11->Dec]
        """
        result = np.zeros((12,))
        for m in xrange(1, 13):
            bool_vector = map(lambda x : x.month == m, self.time)
            indices = np.where(bool_vector)[0]
            result[m - 1] = np.mean(np.array(self.data)[indices])
        return result


    def get_daily_normals(self, start_date = None, end_date = None, stamp_year = 2001):
        """
        :type start_date: datetime.datetime
        :type end_date: datetime.datetime
        :rtype : list , list
        """
        the_date = date(stamp_year, 1, 1)

        day = timedelta(days = 1)
        year_dates = [ ]

        #creat objects for each day of year
        while the_date.year == stamp_year:
            year_dates.append(the_date)
            the_date += day



        if start_date is None:
            start_date = self.time[0]

        if end_date is None:
            end_date = self.time[-1]



        daily_means = []
        for stamp_day in year_dates:
            bool_vector = map(lambda x: x.day == stamp_day.day and
                                        x.month == stamp_day.month and
                                        x >= start_date and
                                        x <= end_date, self.time)

            indices = np.where( bool_vector )[0]
            daily_means.append(np.array(self.data)[indices].mean())

        return year_dates, daily_means


    def get_size(self):
        return len(self.time)



class DateValuePair:
    def __init__(self, date = None, value = None):
        """
        Object for holding corresponding date and value
        pairs
        """
        self.date = date
        self.value = value
        pass


def main():
    #TODO: implement
    pass

if __name__ == "__main__":
    main()
    print "Hello world"
  