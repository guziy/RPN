

from pendulum import Period, Pendulum
from collections import OrderedDict

__author__ = 'huziy'

DEFAULT_SEASON_TO_MONTHS = OrderedDict([
    ("Winter", (1, 2, 12)),
    ("Spring", range(3, 6)),
    ("Summer", range(6, 9)),
    ("Fall", range(9, 12)),
])




class MonthPeriod(object):
    def __init__(self, start_month=1, nmonths=12):

        assert 1 <= start_month <= 12

        self.start_month = start_month
        self.nmonths = nmonths

        self.months = self._get_month_list()



    def _get_month_list(self):
        months = list(range(self.start_month, self.start_month + self.nmonths))
        for i in range(len(months)):
            mi = months[i]

            if mi > 12:
                mi %= 12

                if mi == 0:
                    mi = 12

            months[i] = mi

        return months


    def get_start_and_duration(self):
        return self.start_month, self.nmonths


    def contains(self, month):
        for m in range(self.start_month, self.start_month + self.nmonths):
            m %= 12
            if m == 0:
                m = 12

            if m == month:
                return True

        return False


    def get_season_periods(self, start_year, end_year) -> list:

        plist = []

        for y in range(start_year, end_year + 1):
            start = Pendulum(y, self.start_month, 1)
            end = start.add(months=self.nmonths).subtract(microseconds=1)


            if end.year > end_year:
                continue

            # print(start, end)
            plist.append(
                Period(start, end)
            )

        return plist


    def get_season_interval(self):

        pass

    def get_year_month_to_period_map(self, start_year, end_year):

        """
        generate mapping (year, month) --> period
        :param start_year: 
        :param end_year: 
        :return: 
        """
        res = {}

        for y in range(start_year, end_year + 1):
            start = Pendulum(y, self.start_month, 1)
            end = start.add(months=self.nmonths).subtract(microseconds=1)


            if end.year > end_year:
                continue

            print(start, end)
            p = Period(start, end)

            for s in p.range("months"):
                res[(s.year, s.month)] = p

        return res



DEFAULT_SEASON_TO_MONTHPERIOD = OrderedDict([
    ("DJF", MonthPeriod(12, 3)),
    ("MAM", MonthPeriod(3, 3)),
    ("JJA", MonthPeriod(6, 3)),
    ("SON", MonthPeriod(9, 3))
])
