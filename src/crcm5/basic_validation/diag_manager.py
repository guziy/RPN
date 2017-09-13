import calendar
from collections import OrderedDict
from pathlib import Path

from pendulum import Period
from rpn import level_kinds
from rpn.rpn import RPN

from lake_effect_snow.base_utils import VerticalLevel
from util.seasons_info import MonthPeriod
from util import seasons_info

import numpy as np


class DiagCrcmManager(object):

    def __init__(self, data_dir=""):

        self.data_dir = Path(data_dir)


        self.lons = None
        self.lats = None
        self.projection_params = None
        self.basemap = None

        self._init_grid()

        self.month_folder_map = {}
        self._detect_month_folders()



    def get_basemap(self, **kwargs):
        from rpn.domains.rotated_lat_lon import RotatedLatLon

        rll = RotatedLatLon(**self.projection_params)
        return rll.get_basemap_object_for_lons_lats(lons2d=self.lons, lats2d=self.lats, **kwargs)


    def _init_grid(self):
        for month_dir in self.data_dir.iterdir():
            if not month_dir.is_dir():
                continue

            for f in month_dir.iterdir():
                if f.name.startswith("."):
                    continue

                if f.is_dir():
                    continue


                with RPN(str(f)) as r:
                    assert isinstance(r, RPN)

                    vnames = r.get_list_of_varnames()
                    vname = [v for v in vnames if v not in [">>", "^^", "HY", "CONF", "GSET"]][0]

                    r.get_first_record_for_name(vname)

                    self.lons, self.lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
                    self.projection_params = r.get_proj_parameters_for_the_last_read_rec()


                    # print(r.get_first_record_for_name("GSET"))
                    # print(r.get_first_record_for_name("CONF"))



                    return



    def _detect_month_folders(self):
        for month_dir in self.data_dir.iterdir():

            if not month_dir.is_dir():
                continue


            date_s = month_dir.name.split("_")[-1]
            month, year = int(date_s[-2:]), int(date_s[:-2])

            self.month_folder_map[year, month] = month_dir




    def get_seasonal_means_with_ttest_stats_interp_to(self, lons2d=None, lats2d=None,
                                                      season_to_monthperiod=None, start_year=None, end_year=None):

        #TODO: implement
        pass





    def get_seasonal_means_with_ttest_stats(self, season_to_monthperiod=None, start_year=None, end_year=None,
                                            vname="",
                                            data_file_prefix=None,
                                            vertical_level=VerticalLevel(-1, level_type=level_kinds.ARBITRARY)):
        """
        :param season_to_monthperiod:
        :param start_year:
        :param end_year:
        :return dict(season: [mean, std, nobs])
        """


        season_to_res = OrderedDict()

        levels = None


        for season, season_period in season_to_monthperiod.items():

            assert isinstance(season_period, MonthPeriod)

            seasonal_means = []
            ndays_per_season = []



            for period in season_period.get_season_periods(start_year=start_year, end_year=end_year):

                assert isinstance(period, Period)

                monthly_means = []
                ndays_per_month = []

                for start in period.range("months"):
                    print(season, start)
                    print(self.data_dir)

                    month_dir = self.month_folder_map[start.year, start.month]


                    for data_file in month_dir.iterdir():

                        # check only files with the specified prefix (if the prefix is specified)
                        if data_file_prefix is not None:
                            if not data_file.name.startswith(data_file_prefix):
                                continue

                        # skip files with the variance
                        if data_file.name.endswith("_variance"):
                            continue


                        try:
                            with RPN(str(data_file)) as r:
                                assert isinstance(r, RPN)
                                data = r.get_4d_field(vname, level_kind=vertical_level.level_type)

                                for t, lev_to_field in data.items():


                                    if vertical_level.value == -1:
                                        levels = sorted(lev_to_field) if levels is None else levels
                                        data = np.array([lev_to_field[lev] for lev in levels]).squeeze()
                                    else:
                                        data = lev_to_field[vertical_level.value]

                                    monthly_means.append(data)
                                    ndays_per_month.append(calendar.monthrange(start.year, start.month)[1])
                                    break


                        except Exception as exc:
                            print(exc)


                monthly_means = np.array(monthly_means)
                ndays_per_month = np.array(ndays_per_month)


                # calculate seasonal means
                ndays_per_season.append(ndays_per_month.sum())


                if monthly_means.ndim == 3:
                    seasonal_means.append((monthly_means * ndays_per_month[:, np.newaxis, np.newaxis]).sum(axis=0) / ndays_per_month.sum())
                elif monthly_means.ndim == 4:
                    seasonal_means.append((monthly_means * ndays_per_month[:, np.newaxis, np.newaxis, np.newaxis]).sum(axis=0) / ndays_per_month.sum())
                else:
                    raise NotImplementedError("Cannot handle {}-dimensional data".format(monthly_means.ndim))


            # calculate climatology and ttest params
            seasonal_means = np.array(seasonal_means)
            ndays_per_season = np.array(ndays_per_season)

            mean_clim = (seasonal_means * ndays_per_season[:, np.newaxis, np.newaxis]).sum(axis=0) / ndays_per_season.sum()
            std_clim = (((seasonal_means - mean_clim) ** 2 * ndays_per_season[:, np.newaxis, np.newaxis]).sum(axis=0) / ndays_per_season.sum()) ** 0.5

            season_to_res[season] = [mean_clim, std_clim, len(seasonal_means)]

        return season_to_res






def test():
    manager = DiagCrcmManager(data_dir="/HOME/huziy/skynet3_rech1/CRCM5_outputs/NEI/diags/NEI_WC0.44deg_default/Diagnostics/")
    manager.get_seasonal_means_with_ttest_stats(start_year=1980, end_year=2010, season_to_monthperiod=seasons_info.DEFAULT_SEASON_TO_MONTHPERIOD, vname="TT", data_file_prefix="dm", vertical_level=VerticalLevel(1, level_type=level_kinds.HYBRID))

if __name__ == '__main__':
    test()
