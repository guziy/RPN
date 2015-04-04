from datetime import date, timedelta, datetime
import itertools
import pandas
from pandas.core.frame import DataFrame

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
        the_dict = dict(list(zip(self.time, self.data)))
        return np.array([the_dict[x] for x in the_dates])


    def get_ts_of_dt_means(self, dt = timedelta(days = 1)):
        #TODO: implement
        pass

    def time_slice(self, start_date, end_date):
        bool_vector = np.array( [start_date <= t <= end_date for t in self.time] )

        new_times =  list( filter( lambda t: start_date <= t <= end_date, self.time))
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
        print(end_date)
        while t0 < end_date:
            bool_vector = np.array( [(x.month == t0.month) and
                                                  (x.year == t0.year) for x in self.time] )

            assert np.any(bool_vector), t0
            new_times.append(t0)
            new_data.append(np.mean(self.data[bool_vector]))

            if t0.month + 1 <= 12:
                t0 = t0.replace(month=t0.month + 1)
            else:
                t0 = t0.replace(year=t0.year + 1,month=1)

        print("initial data = from {0} to {1}".format(min(self.data), max(self.data)))
        print("monthly means = from {0} to {1}".format(min(new_data), max(new_data)))
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
        print(end_date)
        while t0 < end_date:
            bool_vector = np.array( [(x.month == t0.month) and
                                                  (x.year == t0.year) for x in self.time] )

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
            bool_vector = np.array( [(x.day == t0.day) and
                                                  (x.month == t0.month) and
                                                  (x.year == t0.year) for x in self.time] )


            assert np.any(bool_vector), t0
            new_times.append(t0)
            new_data.append(np.mean(self.data[bool_vector]))
            t0 += day

        print("initial data = from {0} to {1}".format(min(self.data), max(self.data)))
        print("daily means = from {0} to {1}".format(min(new_data), max(new_data)))
        ts = TimeSeries(data=np.array(new_data), time=new_times)
        ts.metadata = self.metadata
        return ts
        pass

    def get_mean(self, months = range(1,13)):
        """
        returns mean over the months speciifed in the
        months parameter
        """
        bool_vector = [x.month in months for x in self.time]
        indices = np.where(bool_vector)[0]
        return np.mean(self.data[indices])

    def get_monthly_normals(self):
        """
        returns the list of 12 monthly normals corresponding
        to the 12 months [0->Jan, ..., 11->Dec]
        """
        result = np.zeros((12,))
        for m in range(1, 13):
            bool_vector = [x.month == m for x in self.time]
            indices = np.where(bool_vector)[0]
            result[m - 1] = np.mean(np.array(self.data)[indices])
        return result


    def get_daily_normals(self, start_date = None, end_date = None, stamp_year = 2001):
        """
        :type start_date: datetime.datetime
        :type end_date: datetime.datetime
        :rtype : list , list
        """
        self.stamp_day_dates = pandas.DatetimeIndex(start = datetime(stamp_year,1,1), end = date(stamp_year, 12, 31),
            freq = pandas.datetools.offsets.Day())

        if start_date is None:
            start_date = self.time[0]

        if end_date is None:
            end_date = self.time[-1]


        di = pandas.DatetimeIndex(data = self.time)
        df = DataFrame(data = self.data, index = di, columns=["values",])


        df = df.select( lambda d: start_date <= d <= end_date )
        df_mean = df.groupby(by = lambda d: (d.day, d.month)).mean()


        return self.stamp_day_dates, df_mean.ix[[ (d.day, d.month) for d in self.stamp_day_dates] ,"values"]


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
    print("Hello world")
  