from pendulum import Period
import calendar

class CcPeriodsInfo(object):


    CAL_GREGORIAN = "gregorian"
    CAL_365_day = "365_day"
    CAL_360_day = "360_day"

    def __init__(self, cur_period: Period=None, fut_period: Period=None, calendar=CAL_GREGORIAN):
        self.cur_period = cur_period
        self.fut_period = fut_period
        self.calendar = calendar

        self.num_days_for_season_cache = {}

    def get_fut_year_limits(self):
        return self._get_year_limits(self.fut_period)

    def get_cur_year_limits(self):
        return self._get_year_limits(self.cur_period)


    def _get_year_limits(self, period: Period):
        end_year = period.end.year if period.end.month == 12 else period.end.year - 1
        return period.start.year, end_year


    def get_numdays_for_season(self, year, month_list):

        # try touse cache
        key = (year, ) + tuple(month_list)
        if key in self.num_days_for_season_cache:
            return self.num_days_for_season_cache[key]


        res = 0
        for m in month_list:
            _, nd = calendar.monthrange(year, m)

            if self.calendar == self.CAL_365_day and m == 2:
                nd = 28
            elif self.calendar == self.CAL_360_day:
                nd = 30

            res += nd

        self.num_days_for_season_cache[key] = res
        return res