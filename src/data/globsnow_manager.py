import os
from collections import OrderedDict
from datetime import datetime

import numpy as np


# Based on the MonthlyGlobSnowManager.info(), the monthly data is ok to use for winter and spring seasonal means over the 1982-2011 period


def _get_ymonth_from_fname(fname):
    ym = fname.split("_")[-2]
    return int(ym[:-2]), int(ym[-2:])


class MonthlyGlobSnowManager(object):


    def __init__(self, data_folder="/HOME/huziy/skynet3_rech1/obs_data/SWE/GLOBSNOW/monthly", nc_varname="SWE_avg"):
        self.folder = data_folder
        self.nc_varname = nc_varname

        # self.folder should be set before the call
        self._monthdate_to_fpath = self.__map_date_to_file_path()


    def get_seasonal_clim(self, start_year=-np.Inf, end_year=np.Inf, season_to_months=None):

        # season to the climatology field
        res = OrderedDict()

        for season, months in season_to_months.items():
            pass



    def get_seasonal_clim_interpolated_to(self, target_lons=None, target_lats=None, **kwargs):

        seas_means = self.get_seasonal_clim(**kwargs)


    def __map_date_to_file_path(self):
        res = {}
        for fn in os.listdir(self.folder):
            y, m = _get_ymonth_from_fname(fn)
            res[datetime(y, m, 1)] = os.path.join(self.folder, fn)
        return res


    def info(self):
        months = []
        for fn in os.listdir(self.folder):
            y, m = _get_ymonth_from_fname(fn)
            d = datetime(y, m, 1)
            months.append(d)

        months.sort()
        print("Date range {:%Y-%m} ... {:%Y-%m}".format(months[0], months[-1]))
        print("Total number of months: {}".format(len(months)))

        year_2_months = OrderedDict()
        for m in months:
            if m.year not in year_2_months:
                year_2_months[m.year] = []

            year_2_months[m.year].append(m.month)

        print("*" * 20)

        for y, mm in year_2_months.items():
            print("{} months available for {}: {}".format(len(mm), y, mm) + ", " + ("OK!" if len(mm) == 12 else "MISSING SOME!") )

        print("*" * 20)


        season_to_months = OrderedDict(
            [("Winter", {1, 2, 12}),
            ("Spring", {3, 4, 5})])


        season_to_bad_years = OrderedDict([(s, []) for s in season_to_months])
        for season, smonths in season_to_months.items():
            print("Checking {}".format(season))
            for y, mm in year_2_months.items():
                if smonths.intersection(set(mm)) == smonths:
                    print("{} is OK".format(y))
                else:
                    print("for {} -- some missing".format(y))
                    season_to_bad_years[season].append(y)

        print("Season to incomplete years list")
        print(season_to_bad_years)


def test():
    m = MonthlyGlobSnowManager()
    m.info()

if __name__ == '__main__':
    test()