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

