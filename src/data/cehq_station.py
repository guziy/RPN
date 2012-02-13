import itertools

__author__="huziy"
__date__ ="$8 dec. 2010 10:38:26$"

import re
import os
import codecs
from datetime import datetime, timedelta, date
import numpy as np


import application_properties



class Station:
    def __init__(self):
        self.id = None
        self.name = None
        self.longitude = None
        self.latitude = None
        self.drainage_km2 = None
        self.natural = False

        self.dates = []
        self.values = []
        
        self.date_to_value = {}
        

    def get_monthly_normals(self):
        """
        returns the list of 12 monthly normals corresponding
        to the 12 months [0->Jan, ..., 11->Dec]
        """
        result = np.zeros((12,))
        for m in xrange(1, 13):
            bool_vector = map(lambda x : x.month == m, self.dates)
            indices = np.where(bool_vector)[0]
            result[m - 1] = np.mean(np.array(self.values)[indices])
        return result


    def get_daily_normals(self, start_date = None, end_date = None, stamp_year = 2001):
        """
        :type start_date: datetime.datetime
        :type end_date: datetime.datetime
        """
        the_date = date(stamp_year, 1, 1)
        day = timedelta(days = 1)
        year_dates = [ ]

        #creat objects for each day of year
        while the_date.year == stamp_year:
            year_dates.append(the_date)
            the_date += day

        if start_date is None:
            start_date = self.dates[0]

        if end_date is None:
            end_date = self.dates[-1]



        daily_means = []
        for stamp_day in year_dates:
            bool_vector = map(lambda x: x.day == stamp_day.day and
                                        x.month == stamp_day.month and
                                        x >= start_date and
                                        x <= end_date, self.dates)

            indices = np.where( bool_vector )[0]
            daily_means.append(np.array(self.values)[indices].mean())

        return year_dates, daily_means


    def get_value_for_date(self, the_date):
        if len(self.date_to_value) != len(self.dates):
            self.date_to_value = dict(zip(self.dates, self.values))
        return self.date_to_value[the_date]

    def remove_all_observations(self):
        self.dates = []
        self.values = []
        self.date_to_value = {}
        pass


    def delete_data_for_year(self, year):
        to_remove_dates = []
        to_remove_values = []

        for i, d in enumerate(self.dates):
            if d.year == year:
                to_remove_dates.append(d)
                to_remove_values.append(self.values[i])

        for d, v in zip(to_remove_dates, to_remove_values):
            self.dates.remove(d)
            self.values.remove(v)
            if self.date_to_value.has_key(d):
                del self.date_to_value[d]
        assert len(self.dates) == len(self.date_to_value)


    def delete_data_before_year(self, year):
        if self.dates[0].year >= year:
            return

        for the_year in xrange(self.dates[0].year, year):
            self.delete_data_for_year(the_year)


    def delete_data_after_year(self, year):
        if self.dates[-1].year <= year:
            return

        for the_year in xrange(year + 1, self.dates[-1].year + 1):
            self.delete_data_for_year(the_year)



    #returns a dict {date => value}
    #if the data for the year is not continuous returns an empty dict
    def get_continuous_dataseries_for_year(self, year, data_step = timedelta(days = 1)):
        result = {}
        previous_date = None
        for the_date, value in zip(self.dates, self.values):
            if the_date.year > year:
                if previous_date is not None and the_date - previous_date <= data_step:
                    return result
                else:
                    return {}
            elif the_date.year < year:
                continue
            else:
                if previous_date is not None:
                    if the_date - previous_date > data_step:
                        return {}

                result[the_date] = value
                previous_date = the_date

        print len(result)
        return result
        pass

    #here can be a problem
    def get_longest_continuous_series(self, data_step = timedelta(days = 1)):
        series_list = []
        current_series = []
        for d in self.dates:
            if not len(current_series):
                current_series.append(d)
                series_list.append(current_series)
            else:
                prev_date = current_series[-1]  
                if d - prev_date > data_step:
                    current_series = [d]
                    series_list.append(current_series)
                else:
                    current_series.append(d)

        series_list = sorted( series_list, key = lambda x: len(x))

        print map(len, series_list)
        return series_list[-1]




    def remove_record_for_date(self, the_date):
        if the_date in self.dates:
            i = int(self.dates.index(the_date))
            del self.date_to_value[self.dates[i]]
            del self.dates[i]
            del self.values[i]


    def get_timeseries_length(self):
        assert len(self.dates) == len( self.date_to_value ), 'list_len, dict_len = {0},{1}'.format(len(self.dates), len( self.date_to_value ))
        return len(self.dates)

    def parse_from_cehq(self, path, only_natural = False):
        """
        only_natural == False then read all data for all stations,
        otherwize only read data for the natural stations
        """
        f = codecs.open(path, encoding = 'iso-8859-1')
        start_reading_data = False

        dates = []
        values = []


        self.id = re.findall(r"\d+", os.path.basename(path) )[0]
        for line in f:
            line = line.strip()
            line_lower = line.lower().encode('iso-8859-1')

            if line_lower.startswith('station:'):
                [rest, self.name] = line.split(':')

            if 'bassin versant:' in line_lower:
                group = re.findall(r"\d+", line)
                self.drainage_km2 = float(group[0])
                if line_lower.endswith("naturel"):
                    self.natural = True
                else:
                    self.natural = False
                    if only_natural:
                        return




            if '(nad83)' in line_lower:
                groups = re.findall(r"-\d+|\d+", line_lower.replace(' ', '').replace('(nad83)', ''))
                groups = map(float, groups)

                self.latitude = self._get_degrees(groups[0:3])
                self.longitude = self._get_degrees(groups[3:])


            if 'date' in line_lower and 'remarque' in line_lower and 'station' in line_lower:
                start_reading_data = True
                continue

            #read date - value pairs from file
            
            if start_reading_data:
                fields = line.split()
                if len(fields) < 3:
                    continue

                fields = line.split(None, 3)
                dates.append(fields[1])
                values.append(fields[2])


        self.dates = map( lambda t : datetime.strptime(t, '%Y/%m/%d'), dates)
        self.values = map( float, values )
        self.date_to_value = dict(zip(self.dates, self.values))



    def info(self):
        return '%s: lon=%3.1f; lat = %3.1f; drainage(km**2) = %f ' % (self.id,
                                                                self.longitude, self.latitude,
                                                                self.drainage_km2)


    def _get_degrees(self, group):
        """
        Converts group (d,m,s) -> degrees
        """
        [d, m, s] = group
        koef = 1.0 / 60.0
        sign = 1.0 if d >= 0 else -1.0
        return d + sign * koef * m + sign * koef ** 2 * s


    #override hashing methods to use in dictionary
    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        if other is None:
            return False

        return self.id == other.id



    def delete_data_after_date(self, the_date):
        """
        delete values corresponding to the dates later than the_date,
        does not delete the value corresponding to the_date
        """
        vector = map( lambda x: x > the_date, self.dates)

        if True not in vector:
            return

        if False not in vector:
            self.dates = []
            self.values = []
            self.date_to_value = {}
            return

        index = vector.index(True)

        for d in self.dates[index:]:
            del self.date_to_value[d]

        del self.dates[index:], self.values[index:]







    def delete_data_before_date(self, the_date):
        """
        delete values corresponding to the dates earlier than the_date,
        does not delete the value corresponding to the_date
        """
        vector = map( lambda x: x < the_date, self.dates)

        if True not in vector:
            return

        if False not in vector:
            self.dates = []
            self.values = []
            self.date_to_value = {}
            return

        index = vector.index(False)
        for d in self.dates[:index]:
            del self.date_to_value[d]


        del self.dates[:index], self.values[:index]

        pass


def print_info_of(station_ids):
    for the_id in station_ids:
        s = Station()
        path = 'data/cehq_measure_data/%06d_Q.txt' % the_id
        s.parse_from_cehq(path)
        print s.info()
    

def read_station_data(folder = 'data/cehq_measure_data',
                      only_natural = True,
                      start_date = None,
                      end_date = None,
                      selected_ids = None
                      ):
    """
    :rtype: list of data.cehq_station.Station
    if start_date is not None then delete values for t < start_date
    if end_date is not None then delete values for t > end_date

    """
    stations = []
    for file in os.listdir(folder):
        if not '.txt' in file:
            continue
        path = os.path.join(folder, file)
        s = Station()

        s_id = re.findall(r"\d+", os.path.basename(path) )[0]
        if selected_ids is not None:
            if s_id not in selected_ids:
                continue


        s.parse_from_cehq(path, only_natural=only_natural)



        if start_date is not None:
            s.delete_data_before_date(start_date)

        if end_date is not None:
            s.delete_data_after_date(end_date)

        #only save stations with nonzero timeseries length
        if s.get_timeseries_length():
            if only_natural:
                if s.natural:
                    stations.append(s)
            else:
                stations.append(s)


    return stations




if __name__ == "__main__":
    application_properties.set_current_directory()

    station_ids = [104001, 103715, 93801, 93806, 81006, 92715, 61502, 80718, 42607, 40830]
    print_info_of(station_ids)



    s = Station()
    s.parse_from_cehq('data/cehq_measure_data/051004_Q.txt')
    data = s.get_continuous_dataseries_for_year(1970)
#    for date in sorted(data.keys()):
#        print date, '-->', data[date]

    print np.max(s.values)
    print np.max(s.dates)
    print "Hello World"
