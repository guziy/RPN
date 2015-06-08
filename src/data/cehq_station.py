import itertools
import pickle
import pandas

__author__ = "huziy"
__date__ = "$8 dec. 2010 10:38:26$"

import re
import os
import codecs
from datetime import datetime, timedelta, date
import time
import numpy as np

import application_properties


class Station:
    def __init__(self):
        self.source = "Unknown"
        self.id = None
        self.name = None
        self.longitude = None
        self.latitude = None
        self.drainage_km2 = None
        self.natural = False

        self.dates = []
        self.values = []

        self.date_to_value = {}

        # daily climatology of mean swe upstream (as seen by the model) to the station
        # ideally pandas.Timeseries?
        self.mean_swe_upstream_daily_clim = None
        self.mean_temp_upstream_monthly_clim = None
        self.mean_prec_upstream_monthly_clim = None

        ##specifically for the GRDC stations
        self.grdc_monthly_clim_min = None
        self.grdc_monthly_clim_mean = None
        self.grdc_monthly_clim_max = None

        self.river_name = ""
        self._complete_years = None


    def get_mean_value(self):
        return np.mean(self.values)

    def get_monthly_normals(self):
        """
        returns the list of 12 monthly normals corresponding
        to the 12 months [0->Jan, ..., 11->Dec]
        return None if there is even a single month for which there is no data
        """
        result = np.zeros((12,))
        for m in range(1, 13):
            bool_vector = [x.month == m for x in self.dates]
            indices = np.where(bool_vector)[0]

            if not len(indices): return None

            result[m - 1] = np.mean(np.array(self.values)[indices])
        return result


    def get_daily_normals(self, start_date=None, end_date=None, stamp_year=2001):
        """
        :type start_date: datetime.datetime
        :type end_date: datetime.datetime
        """
        the_date = date(stamp_year, 1, 1)
        day = timedelta(days=1)
        year_dates = []

        # creat objects for each day of year
        while the_date.year == stamp_year:
            year_dates.append(the_date)
            the_date += day

        if start_date is None:
            start_date = self.dates[0]

        if end_date is None:
            end_date = self.dates[-1]

        daily_means = []
        for stamp_day in year_dates:
            bool_vector = [x.day == stamp_day.day and
                           x.month == stamp_day.month and
                           start_date <= x <= end_date for x in self.dates]

            indices = np.where(bool_vector)[0]
            if not len(indices): return None, None
            daily_means.append(np.array(self.values)[indices].mean())

        return year_dates, np.array(daily_means)


    def get_value_for_date(self, the_date):
        if len(self.date_to_value) != len(self.dates):
            self.date_to_value = dict(list(zip(self.dates, self.values)))
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
            if d in self.date_to_value:
                del self.date_to_value[d]
        assert len(self.dates) == len(self.date_to_value)


    def delete_data_before_year(self, year):
        if self.dates[0].year >= year:
            return

        for the_year in range(self.dates[0].year, year):
            self.delete_data_for_year(the_year)


    def delete_data_after_year(self, year):
        if self.dates[-1].year <= year:
            return

        for the_year in range(year + 1, self.dates[-1].year + 1):
            self.delete_data_for_year(the_year)


    # returns a dict {date => value}
    # if the data for the year is not continuous returns an empty dict
    def get_continuous_dataseries_for_year(self, year, data_step=timedelta(days=1)):
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

        print(len(result))
        return result
        pass

    #here can be a problem
    def get_longest_continuous_series(self, data_step=timedelta(days=1)):
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

        series_list = sorted(series_list, key=lambda x: len(x))

        print(list(map(len, series_list)))
        return series_list[-1]


    def remove_record_for_date(self, the_date):
        if the_date in self.dates:
            i = int(self.dates.index(the_date))
            del self.date_to_value[self.dates[i]]
            del self.dates[i]
            del self.values[i]


    def get_timeseries_length(self):
        assert len(self.dates) == len(self.date_to_value), 'list_len, dict_len = {0},{1}'.format(len(self.dates), len(
            self.date_to_value))
        return len(self.dates)

    def parse_from_cehq(self, path, only_natural=False):
        """
        only_natural == False then read all data for all stations,
        otherwize only read data for the natural stations
        """
        f = codecs.open(path, encoding='iso-8859-1')
        start_reading_data = False

        dates = []
        values = []

        self.id = re.findall(r"\d+", os.path.basename(path))[0]
        for line in f:
            line = line.strip()
            line_lower = line.lower()

            if line_lower.startswith('station:'):
                [rest, self.name] = line.split(':')
                self.source = "CEHQ"

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
                groups = list(map(float, groups))

                self.latitude = _get_degrees(groups[0:3])
                self.longitude = _get_degrees(groups[3:])

            if 'date' in line_lower and 'remarque' in line_lower and 'station' in line_lower:
                start_reading_data = True
                continue

            #read date - value pairs from file

            if start_reading_data:
                fields = line.split()
                if len(fields) < 3:
                    continue

                try:
                    float(fields[2])
                except ValueError:
                    continue

                dates.append(fields[1])
                values.append(fields[2])

        self.dates = [datetime.strptime(t, '%Y/%m/%d') for t in dates]
        self.values = list(map(float, values))
        self.date_to_value = dict(list(zip(self.dates, self.values)))


    def info(self):
        return '%s: lon=%3.1f; lat = %3.1f; drainage(km**2) = %f ' % (self.id,
                                                                      self.longitude, self.latitude,
                                                                      self.drainage_km2)


    # override hashing methods to use in dictionary
    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        if other is None:
            return False

        return self.id == other.id


    @classmethod
    def get_stamp_days(cls, stamp_year):
        """
        returns stampdates for the year
        """
        dt = timedelta(days=1)
        return [datetime(stamp_year, 1, 1) + i * dt for i in range(365)]


    def delete_data_after_date(self, the_date):
        """
        delete values corresponding to the dates later than the_date,
        does not delete the value corresponding to the_date
        """
        vector = [x > the_date for x in self.dates]

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
        vector = [x < the_date for x in self.dates]

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

    def passes_rough_continuity_test(self, start_date, end_date):
        nyears = end_date.year - start_date.year + 1
        nentries = sum([int(start_date <= t <= end_date) for t in self.dates])
        return nentries >= 365 * nyears

    def get_list_of_complete_years(self):
        """
        assumes that the observed data frequency is daily
        """

        if self._complete_years is not None:
            return self._complete_years

        years = []

        years_all = np.array([d.year for d in self.dates])
        years_unique = np.unique(years_all)

        for y in years_unique:
            count = np.array(years_all == y, dtype=bool).astype(int).sum()
            if count >= 365:
                years.append(y)

        self._complete_years = years
        return years


    def get_daily_climatology_for_complete_years(self, stamp_dates=None, years=None):
        if stamp_dates is None:
            stamp_year = 2001
            stamp_dates = Station.get_stamp_days(stamp_year)

        if years is None:
            years = self.get_list_of_complete_years()

        vals = []
        all_data = np.array(self.values)

        year_mask = np.array([d.year in years for d in self.dates])
        for d in stamp_dates:
            mask = [1 if d.month == x.month and d.day == x.day else 0 for x in self.dates]
            x = np.array(mask) * all_data * year_mask
            vals.append(np.mean(x[x > 0]))

        return stamp_dates, vals


    def get_daily_climatology_for_complete_years_with_pandas(self, stamp_dates=None, years=None):
        assert stamp_dates is not None
        assert years is not None

        df = pandas.DataFrame(data=self.values, index=self.dates, columns=["values", ])

        df["year"] = df.index.map(lambda the_date: the_date.year)

        df = df.select(lambda d: d.year in years and not (d.month == 2 and d.day == 29))

        if len(df) == 0:
            return None, None

        stamp_year = stamp_dates[0].year


        daily_clim = df.groupby(by=lambda the_date: datetime(stamp_year, the_date.month, the_date.day)).mean()

        print(daily_clim.head())
        pickle.dump(daily_clim, open("/home/huziy/daily_clim.bin", "wb"))
        pickle.dump(stamp_dates, open("/home/huziy/stamp_dates.bin", "wb"))

        print("The type of the first element in stamp_dates is {}".format(type(stamp_dates[0])))

        vals = [daily_clim.ix[d, "values"] for d in stamp_dates]


        return stamp_dates, vals


    def __str__(self):
        return "Gauge station ({0}): {1} at ({2},{3}), accum. area is {4} km**2".format(self.id, self.name,
                                                                                        self.longitude,
                                                                                        self.latitude,
                                                                                        self.drainage_km2)


    def parse_from_hydat(self, path):
        f = open(path)

        lines = f.readlines()

        read_data_flag = False
        for line in lines:
            line = line.strip()

            if line == "":
                continue
            if not read_data_flag:
                line = line.lower()
                read_data_flag = "id" in line and "datatype" in line and "date" in line
                read_data_flag = "value" in line and read_data_flag

                if line.startswith("superf"):
                    sarea = re.findall("\\d+", line)

                    self.drainage_km2 = float(sarea[0])
                elif line.startswith("longitude") or line.startswith("latitude"):
                    groups = re.findall("\\d+", line)

                    degs, mins, secs = [float(g) for g in groups]
                    coef = -1 if line.endswith("w") or line.endswith("s") else 1
                    if line.startswith("longitude"):
                        self.longitude = (degs + mins / 60.0 + secs / 3600.0) * coef
                    else:
                        self.latitude = (degs + mins / 60.0 + secs / 3600.0) * coef




            else:
                fields = line.split(",")
                if self.id is None:
                    self.id = fields[0].strip()

                date = datetime.strptime(fields[2].strip(), "%Y/%m/%d")
                val = float(fields[3].strip())

                self.dates.append(date)
                self.values.append(val)

        self.date_to_value = dict(list(zip(self.dates, self.values)))

    def read_data_from_hydat_db_results(self, data, start_date=None, end_date=None, variable="streamflow"):
        """
        read data from results of request to hydat database

        :param variable can be either streamflow or level

        :param data: list of dictionaries the values of interest are under the following keys
            YEAR, MONTH, NO_DAYS, FLOW1, FLOW2, FLOW3, ..., FLOW31
            NO_DAYS - number of days in a given month
        """
        df_list = []
        for row in data:
            # Extracts data from a row, one month of data per row
            ndays = row["NO_DAYS"]
            year = row["YEAR"]
            month = row["MONTH"]

            month_dates = [datetime(year, month, i) for i in range(1, ndays + 1)]

            if variable.lower() == "streamflow":
                prefix = "FLOW"
            elif variable.lower() == "level":
                prefix = "LEVEL"
            else:
                raise Exception("Unknown variable: {0}".format(variable))

            month_vals = [row["{0}{1}".format(prefix, i)] for i in range(1, ndays + 1)]

            df_month = pandas.DataFrame(data=month_vals, index=month_dates, columns=["value"])
            df_list.append(df_month)

        df = pandas.concat(df_list, verify_integrity=True)
        df.sort(inplace=True)
        if start_date is not None:
            df = df.select(lambda d: start_date <= d <= end_date)

        if end_date is not None:
            df = df.select(lambda d: d <= end_date)

        if not len(df):
            self.dates = []
            self.values = []
            self.date_to_value = {}
            return

        self.dates = df.index
        self.values = df.values.flatten()
        self.date_to_value = dict(list(zip(self.dates, self.values)))

        pass


def _get_degrees(group):
    """
    Converts group (d,m,s) -> degrees
    """
    [d, m, s] = group
    koef = 1.0 / 60.0
    sign = 1.0 if d >= 0 else -1.0
    return d + sign * koef * m + sign * koef ** 2 * s


def print_info_of(station_ids):
    for the_id in station_ids:
        s = Station()
        path = 'data/cehq_measure_data/%06d_Q.txt' % the_id
        s.parse_from_cehq(path)
        print(s.info())


def _get_station_for_id(the_id, st_list):
    return next(filter(lambda x: x.id == the_id, st_list))


def read_station_data(folder='data/cehq_measure_data',
                      only_natural=True,
                      start_date=None,
                      end_date=None,
                      selected_ids=None, min_number_of_complete_years=3):
    """
    :return type: list of data.cehq_station.Station
    if start_date is not None then delete values for t < start_date
    if end_date is not None then delete values for t > end_date

    """
    stations = []
    for the_file in os.listdir(folder):
        if not the_file.endswith(".txt"):
            continue
        path = os.path.join(folder, the_file)
        s = Station()

        s_id = re.findall(r"\d+", os.path.basename(path))[0]
        if selected_ids is not None:
            if s_id not in selected_ids:
                continue

        s.parse_from_cehq(path, only_natural=only_natural)

        if start_date is not None:
            s.delete_data_before_date(start_date)

        if end_date is not None:
            s.delete_data_after_date(end_date)


        # If there is less than 3 years of continuous data, discard the station
        if len(s.get_list_of_complete_years()) <= min_number_of_complete_years:
            continue

        # only save stations with nonzero timeseries length
        if s.get_timeseries_length():
            if only_natural:
                if s.natural:
                    stations.append(s)
            else:
                stations.append(s)

    if selected_ids is not None:
        stations = [_get_station_for_id(x, stations) for x in selected_ids]

    print("Got {0} stations from {1}".format(len(stations), folder))
    return stations


def read_hydat_station_data(folder_path="", start_date=None, end_date=None):
    """
    Read files downloaded from EC website (csv)
    """
    stations = []
    for the_file in os.listdir(folder_path):
        path = os.path.join(folder_path, the_file)
        s = Station()

        s.parse_from_hydat(path)
        if start_date is not None:
            s.delete_data_before_date(start_date)

        if end_date is not None:
            s.delete_data_after_date(end_date)

        # only save stations with nonzero timeseries length
        if s.get_timeseries_length():
            stations.append(s)

    return stations


def _prep_line(line):
    n = len(line)

    counter = 0
    new_line = ""
    for i in range(n):
        counter += int(line[i] == "\"")
        if line[i] == " " and counter % 2 == 1:
            new_line += "_"
        else:
            new_line += line[i]

    return new_line


def read_grdc_stations(st_id_list=None, data_file_patt="/skynet3_rech1/huziy/GRDC_streamflow_data/Data{0}.txt",
                       descriptor_file_path="/skynet3_rech1/huziy/GRDC_all_stations/GRDC663Sites.txt"):
    """
    extracts station coordinates and monthly climatology
    """
    res = []

    descr_file = open(descriptor_file_path)
    lines = descr_file.readlines()
    descr_file.close()

    fields = lines[0].split()
    print(fields)
    print(fields[3], fields[-2], fields[-1], fields[4], fields[6])

    for line in lines[1:]:
        line = line.strip()
        if line == "":
            continue

        line = _prep_line(line)

        fields = line.split()
        the_id = fields[3].strip()

        if the_id not in st_id_list:
            continue

        lon = float(fields[-2])
        lat = float(fields[-1])

        s = Station()
        s.id = the_id
        s.longitude = lon
        s.latitude = lat
        s.river_name = fields[5].replace("_", " ").replace("\"", "")
        s.drainage_km2 = float(fields[20])

        # print "found {0}".format(the_id) , s.river_name
        # print "DA(GRDC) = {0}; DA(STNCatchment) = {1}".format(s.drainage_km2, fields[20])
        # load data
        # load min, mean and max
        data_path = data_file_patt.format(the_id)

        s.grdc_monthly_clim_min = []
        s.grdc_monthly_clim_mean = []
        s.grdc_monthly_clim_max = []

        lines = open(data_path).readlines()
        for line in lines[1:]:
            fields = line.split()
            s.grdc_monthly_clim_max.append(float(fields[-2]))
            s.grdc_monthly_clim_mean.append(float(fields[-1]))
            s.grdc_monthly_clim_min.append(float(fields[-3]))

        res.append(s)

    return res

    pass


def load_from_hydat_db(path="/home/huziy/skynet3_rech1/hydat_db/Hydat.sqlite",
                       natural=True,
                       province="QC", start_date=None, end_date=None, datavariable="streamflow",
                       min_drainage_area_km2=None, selected_ids=None):
    """
    loads stations from sqlite db

    :param datavariable can be "streamflow" or "level"

    """
    import sqlite3


    assert natural

    # cache_file = "hydat_stations_{0}_{1}.cache".format("natural" if natural else "regulated", province)
    # os.remove(cache_file)
    #    if os.path.isfile(cache_file):
    #        return pickle.load(open(cache_file))

    province = province.upper()

    # shortcuts for table names
    stations_table = "STATIONS"
    station_regulation_table = "STN_REGULATION"

    # shortcuts for field names
    station_number_field = "STATION_NUMBER"
    province_field = "PROV_TERR_STATE_LOC"
    lon_field = "LONGITUDE"
    lat_field = "LATITUDE"

    regulation_table = "STN_REGULATION"

    if datavariable.lower() == "streamflow":
        daily_var_table = "DLY_FLOWS"  # streamflow is in m**3/s
    elif datavariable.lower() == "level":
        daily_var_table = "DLY_LEVELS"
    else:
        raise NotImplementedError("datavariable = {0} is not implemented yet".format(datavariable))

    tables_of_interest = [stations_table, regulation_table, daily_var_table]

    connect = sqlite3.connect(path)
    assert isinstance(connect, sqlite3.Connection)
    connect.row_factory = sqlite3.Row

    cur = connect.cursor()
    assert isinstance(cur, sqlite3.Cursor)

    cur.execute("SELECT * FROM Version;")
    for the_row in cur:
        print(list(the_row.keys()))
        print("using hydat version {0} generated on " \
              "{1}".format(the_row["Version"], datetime.fromtimestamp(the_row["Date"] / 1000.0)))

    cur.execute("SELECT name FROM sqlite_master WHERE type = 'table';")

    table_names = cur.fetchall()

    print(table_names)

    for table_data in table_names:
        # print "Table: {0}".format(table_data["name"])

        assert isinstance(table_data, sqlite3.Row)
        # print table_data.keys()
        # print table_data["name"]

        if table_data["name"] not in tables_of_interest:
            continue  # skip tables which are not interesting


        # determine table layout
        cur.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?;", (table_data["name"],))

        scheme = cur.fetchone()
        print(scheme)
        print("++++" * 20)


    # create station objects using data from sqlite db


    # select stations in quebec region which are not regulated
    query = "select * from {0} join {1} on {0}.STATION_NUMBER = {1}.STATION_NUMBER" \
            " where ({0}.{2}=? or {0}.{2}=?) and {1}.REGULATED={3};".format(stations_table,
                                                                            station_regulation_table,
                                                                            province_field,
                                                                            int(not natural))
    print("query = {0}".format(query))
    cur.execute(query, (province, province.lower()))

    data = cur.fetchall()

    print(list(data[0].keys()))

    print("Fetched the following station: ")
    print("There are {0} non-regulated stations in {1}.".format(len(data), province))

    # the_row = cur.fetchone()

    stations = []
    for the_row in data:
        s = Station()
        s.source = "HYDAT"
        s.longitude = the_row["LONGITUDE"]
        s.latitude = the_row["LATITUDE"]
        s.id = the_row["STATION_NUMBER"]
        s.name = the_row["STATION_NAME"]
        s.drainage_km2 = the_row["DRAINAGE_AREA_GROSS"]

        if selected_ids is not None:
            if s.id not in selected_ids:
                continue

        # Skip the stations without related infoormation
        if s.drainage_km2 is None:
            continue

        if (min_drainage_area_km2 is not None) and (min_drainage_area_km2 >= s.drainage_km2):
            continue

        # read streamflows for the station
        query = "select * from {0} where STATION_NUMBER = ?".format(daily_var_table)
        cur.execute(query, (s.id, ))

        data_for_station = cur.fetchall()
        if len(data_for_station) < 365:  # there is no way it can have at least one complete year
            # skip the stations with no data
            continue

        s.read_data_from_hydat_db_results(data_for_station, start_date=start_date,
                                          end_date=end_date, variable=datavariable)

        if len(s.get_list_of_complete_years()) < 10:
            # also ignore the stations with less than 10 complete years of data
            continue

        stations.append(s)

    # print the_row[province_field]
    connect.close()
    if len(stations) == 0:
        print("Warning: could not find acceptable stations for hydat in {0} region".format(province))
    return stations
    # pickle.dump(stations, open(cache_file, mode="w"))


if __name__ == "__main__":
    application_properties.set_current_directory()

    station_ids = [104001, 103715, 93801, 93806, 81006, 92715, 61502, 80718, 42607, 40830]
    # print_info_of(station_ids)



    # s = Station()
    #s.parse_from_cehq('data/cehq_measure_data/051004_Q.txt')
    #data = s.get_continuous_dataseries_for_year(1970)
    #    for date in sorted(data.keys()):
    #        print date, '-->', data[date]

    #s.parse_from_hydat("/home/huziy/skynet3_rech1/HYDAT/daily_streamflow_02OJ007.csv")

    #s.get_daily_climatology_for_complete_years()
    #print np.max(s.values)
    #print np.max(s.dates)
    t0 = time.clock()
    load_from_hydat_db(province='ON', start_date=datetime(1979, 1, 1), end_date=datetime(1988, 12, 31))
    print("Execution time is: {0} seconds".format(time.clock() - t0))
    #slist = read_grdc_stations(st_id_list=["2903430", "2909150", "2912600", "4208025"],
    #    descriptor_file_path="/skynet3_rech1/huziy/GRDC_all_stations/GRDC663Sites.txt")
    #
    #for s in slist:
    #    assert isinstance(s, Station)
    #    print s.drainage_km2

    print("Hello World")
