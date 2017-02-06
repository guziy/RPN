from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
import pandas as pd


class BaseDataManager(object):

    def __init__(self):
        self.kdtree = None



    def get_daily_climatology_fields(self, start_year=None, end_year=None):
        """
        :return pandas DataFrame of daily mean climatologies
        :param start_year:
        :param end_year:
        :raise NotImplementedError:
        """
        raise NotImplementedError()



    def get_seasonal_fields(self, start_year: int = -np.Inf, end_year: int = np.Inf, months: list = range(1, 13)) -> pd.Panel:
        """
        :param months: to define a season
        :return a pandas panel with a year as a major_axis
        :param start_year:
        :param end_year:
        """
        raise NotImplementedError()



    def get_daily_clim_fields_aggregated_and_interpolated_to(self, start_year=None, end_year=None,
                                              lons_target=None, lats_target=None, n_agg_x=1, n_agg_y=1):
        """
        Aggregate fields to the desired resolution prior to interpolation
        :param start_year:
        :param end_year:
        :param lons_target:
        :param lats_target:
        """
        # Return 365 fields
        df = self.get_daily_climatology_fields(start_year=start_year, end_year=end_year)

        assert isinstance(df, pd.Panel)

        lons1d, lats1d = lons_target.flatten(), lats_target.flatten()
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons1d, lats1d)

        dists, indices = self.kdtree.query(list(zip(xt, yt, zt)), k=n_agg_x * n_agg_y)

        clim_fields = [
            df.loc[:, day, :].values.flatten()[indices].mean(axis=1).reshape(lons_target.shape) for day in df.major_axis
        ]
        clim_fields = np.asarray(clim_fields)
        clim_fields = np.ma.masked_where(np.isnan(clim_fields), clim_fields)
        return df.major_axis, clim_fields



    def get_daily_clim_fields_interpolated_to(self, start_year=None, end_year=None,
                                              lons_target=None, lats_target=None):
        # Return 365 fields
        df = self.get_daily_climatology_fields(start_year=start_year, end_year=end_year)

        assert isinstance(df, pd.Panel)

        lons1d, lats1d = lons_target.flatten(), lats_target.flatten()
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons1d, lats1d)

        dists, indices = self.kdtree.query(list(zip(xt, yt, zt)))

        clim_fields = [
            df.loc[:, day, :].values.flatten()[indices].reshape(lons_target.shape) for day in df.major_axis
        ]
        clim_fields = np.asarray(clim_fields)
        clim_fields = np.ma.masked_where(np.isnan(clim_fields), clim_fields)
        return df.major_axis, clim_fields


    def get_seasonal_fields_interpolated_to(self, start_year=-np.Inf, end_year=np.Inf, lons_target=None, lats_target=None,
                                            months=range(1, 13)):

        df = self.get_seasonal_fields(start_year=start_year, end_year=end_year, months=list(months))

        assert isinstance(df, pd.Panel)

        lons1d, lats1d = lons_target.flatten(), lats_target.flatten()
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons1d, lats1d)

        dists, indices = self.kdtree.query(list(zip(xt, yt, zt)))

        mean_fields = [
            df.loc[:, year, :].values.flatten()[indices].reshape(lons_target.shape) for year in df.major_axis
            ]
        mean_fields = np.asarray(mean_fields)
        mean_fields = np.ma.masked_where(np.isnan(mean_fields), mean_fields)
        return df.major_axis, mean_fields

