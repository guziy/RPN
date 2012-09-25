from datetime import date, timedelta, datetime
import itertools

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



        self.stamp_day_dates = None
        pass

    def get_data_for_dates(self, the_dates):
        """
        Used to select data for the same dates as for obsevations
        """
        the_dict = dict(zip(self.time, self.data))
        return np.array(map(lambda x: the_dict[x], the_dates))


    def get_ts_of_dt_means(self, dt = timedelta(days = 1)):
        #TODO: implement
        pass

    def time_slice(self, start_date, end_date):
        bool_vector = np.array( map(lambda t: start_date <= t <= end_date, self.time) )

        new_times =  list( itertools.ifilter( lambda t: start_date <= t <= end_date, self.time))
        #print bool_vector
        new_data = np.array(self.data)[bool_vector]
        ts = TimeSeries(time=new_times, data=new_data)
        ts.metadata = self.metadata
        return ts


    def get_ts_of_monthly_means(self):
        """
        returns Timeseries obt containing daily means
        """
        new_times = []
        new_data = []
        t0 = self.time[0]
        t0 = datetime(t0.year, t0.month, 1)

        end_date = self.time[-1]
        end_date = datetime(end_date.year, end_date.month, end_date.day)
        if end_date.day != 1:
            if end_date.month + 1 <= 12:
                end_date = end_date.replace(month=end_date.month + 1, day=1)
            else:
                end_date = end_date.replace(year=end_date.year + 1, month=1, day=1)

        self.data = np.array(self.data)
        print end_date
        while t0 < end_date:
            bool_vector = np.array( map(lambda x: (x.month == t0.month) and
                                                  (x.year == t0.year), self.time) )

            assert np.any(bool_vector), t0
            new_times.append(t0)
            new_data.append(np.mean(self.data[bool_vector]))

            if t0.month + 1 <= 12:
                t0 = t0.replace(month=t0.month + 1)
            else:
                t0 = t0.replace(year=t0.year + 1,month=1)

        print "initial data = from {0} to {1}".format(min(self.data), max(self.data))
        print "monthly means = from {0} to {1}".format(min(new_data), max(new_data))
        ts = TimeSeries(data=np.array(new_data), time=new_times)
        ts.metadata = self.metadata
        return ts

    def get_ts_of_monthly_integrals_in_time(self):
        """
        returns Timeseries obt containing daily means,
        Note: the result is not multiplied by timestep
        """
        new_times = []
        new_data = []
        t0 = self.time[0]
        t0 = datetime(t0.year, t0.month, 1)

        end_date = self.time[-1]
        if end_date.day != 1:
            if end_date.month + 1 <= 12:
                end_date = end_date.replace(month=end_date.month + 1, day=1)
            else:
                end_date = end_date.replace(year=end_date.year + 1, month=1, day=1)

        self.data = np.array(self.data)
        print end_date
        while t0 < end_date:
            bool_vector = np.array( map(lambda x: (x.month == t0.month) and
                                                  (x.year == t0.year), self.time) )

            assert np.any(bool_vector), t0
            new_times.append(t0)
            new_data.append(np.sum(self.data[bool_vector]))

            if t0.month + 1 <= 12:
                t0 = t0.replace(month=t0.month + 1)
            else:
                t0 = t0.replace(year=t0.year + 1,month=1)

        ts = TimeSeries(data=np.array(new_data), time=new_times)
        ts.metadata = self.metadata
        return ts



    def get_ts_of_daily_means(self):
        """
        returns Timeseries obt containing daily means
        """
        day = timedelta(days = 1)

        new_times = []
        new_data = []
        t0 = self.time[0]
        t0 = datetime(t0.year, t0.month, t0.day)

        self.data = np.array(self.data)
        while t0 <= self.time[-1]:
            bool_vector = np.array( map(lambda x: (x.day == t0.day) and
                                                  (x.month == t0.month) and
                                                  (x.year == t0.year), self.time) )


            assert np.any(bool_vector), t0
            new_times.append(t0)
            new_data.append(np.mean(self.data[bool_vector]))
            t0 += day

        print "initial data = from {0} to {1}".format(min(self.data), max(self.data))
        print "daily means = from {0} to {1}".format(min(new_data), max(new_data))
        ts = TimeSeries(data=np.array(new_data), time=new_times)
        ts.metadata = self.metadata
        return ts
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

        self.stamp_day_dates = year_dates

        if start_date is None:
            start_date = self.time[0]

        if end_date is None:
            end_date = self.time[-1]



        daily_means = []
        for stamp_day in year_dates:
            bool_vector = map(lambda x: x.day == stamp_day.day and
                                        x.month == stamp_day.month and
                                        start_date <= x <= end_date, self.time)

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
  