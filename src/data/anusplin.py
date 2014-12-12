from datetime import datetime, timedelta
from multiprocessing import Pool
import time
from netCDF4 import Dataset, MFDataset
import os
import itertools
import pickle
from mpl_toolkits.basemap import Basemap
import numpy as np
from pandas.core.panel import Panel
from scipy.spatial.ckdtree import cKDTree
from util.geo import lat_lon

__author__ = 'huziy'
import pandas as pd

# precipitation data are in mm/day
# the temperature is in celsius

# Did not use MFDataset with this data, since not sure if it can handle the peculiar time in days


class AnuSplinManager:
    def __init__(self,
                 folder_path="/home/huziy/skynet1_rech3/anusplin_links",
                 variable="pcp",
                 file_name_preifx="ANUSPLIN_latlon_"):

        self.lons2d = None
        self.lats2d = None
        self.kdtree = None

        self.nc_varname = None
        if variable == "pcp":
            self.nc_varname = "daily_precipitation_accumulation"
        elif variable == "stmx":
            self.nc_varname = "daily_maximum_temperature"
        elif variable == "stmn":
            self.nc_varname = "daily_minimum_temperature"
        else:
            raise Exception("Unknown variable: {0}".format(variable))

        self.folder_path = folder_path
        self.fname_format = "{0}{1}_%Y_%m.nc".format(file_name_preifx, variable)
        self._read_lon_lats()
        self.name = "ANUSPLIN"
        pass

    def _read_lon_lats(self):
        """
        Read the lons and lats, 2d from the first file
        """
        the_date = datetime(1980, 1, 1)
        fpath = os.path.join(self.folder_path, the_date.strftime(self.fname_format))
        ds = Dataset(fpath)
        self.lons2d = ds.variables["lon"][:].transpose()
        self.lats2d = ds.variables["lat"][:].transpose()

        x, y, z = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
        self.kdtree = cKDTree(zip(x, y, z))

        ds.close()

    def _get_year(self, fname):
        return datetime.strptime(fname, self.fname_format)

    def get_daily_climatology_fields(self, start_year=1980, end_year=1988):
        """
        :rtype : pandas.DataFrame
        :param start_year: start year of the averaging period
        :param end_year: end year of the averaging period
        :return pandas DataFrame of daily mean climatologies
        """
        stamp_year = 2001

        cache_file = "anusplin_daily_climatology_{0}_{1}_{2}.cache.hdf".format(start_year, end_year, self.nc_varname)
        if os.path.isfile(cache_file):
            store = pd.HDFStore(cache_file)
            df = store.get("dailyClimFrame")
            store.close()
            return df

        import calendar


        # open files
        sorted_keys = []
        day_month_to_nc_path = {}
        for month in range(1, 13):
            wkday, ndays = calendar.monthrange(stamp_year, month)
            for day in range(1, ndays + 1):
                key = (month, day)
                day_month_to_nc_path[key] = []
                sorted_keys.append(key)

        for the_year in range(start_year, end_year + 1):
            for month in range(1, 13):
                fname = datetime(the_year, month, 1).strftime(self.fname_format)
                fpath = os.path.join(self.folder_path, fname)
                ds = Dataset(fpath)

                month_days = ds.variables["time"][:] if month != 2 else range(1, 29)
                ds.close()
                for day in month_days:
                    key = (month, int(day))
                    day_month_to_nc_path[key].append(fpath)

        assert len(sorted_keys) == 365
        times = [datetime(stamp_year, the_key[0], the_key[1]) for the_key in sorted_keys]

        pool = Pool(processes=20)

        path_lists = [day_month_to_nc_path[key] for key in sorted_keys]
        day_of_month_list = [key[1] for key in sorted_keys]
        varname_list = [self.nc_varname] * len(path_lists)



        # daily_clim_fields = [
        #    np.asarray([ds.variables[self.nc_varname][day - 1, :, :] for ds in ds_list]).mean(axis=0)
        #    for day, ds_list in zip(day_list, ds_lists)
        #]

        daily_clim_fields = pool.map(_reader, zip(path_lists, day_of_month_list, varname_list))
        daily_clim_fields = np.asarray(daily_clim_fields)
        daily_clim_fields = np.ma.masked_where(np.isnan(daily_clim_fields), daily_clim_fields)

        ni, nj = daily_clim_fields[0].shape
        panel = pd.Panel(data=daily_clim_fields.transpose((1, 0, 2)), major_axis=times, minor_axis=range(nj),
                         items=range(ni))

        store = pd.HDFStore(cache_file)
        store.put("dailyClimFrame", panel)
        store.close()

        return panel


    def getMeanFieldForMonths(self, months=None, start_year=1979, end_year=1988):
        if months is None:
            months = range(1, 13)

        df = self.get_daily_climatology_fields(start_year=start_year, end_year=end_year)

        assert isinstance(df, Panel)

        df_monthly = df.select(lambda d: d.month in months, axis=1)

        df_mean = df_monthly.apply(np.mean)

        assert isinstance(df_mean, pd.DataFrame)
        print df_mean.values.shape

        field = df_mean.values
        return np.ma.masked_where(np.isnan(field), field)


    def get_longest_rain_event_durations(self):
        """
        Get the durations in days of precipitation events
        """
        # TODO: implement
        pass


    def getMeanFieldForMonthsInterpolatedTo(self, months=None, lonstarget=None, latstarget=None,
                                            start_year=None, end_year=None):
        """


        :type lonstarget: object
        :param months:
        :param lonstarget:
        :param latstarget:
        :param start_year:
        :param end_year:
        :return: the field in mm/day
        """
        mean_field = self.getMeanFieldForMonths(months=months, start_year=start_year, end_year=end_year)

        lons1d, lats1d = lonstarget.flatten(), latstarget.flatten()
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons1d, lats1d)

        dists, indices = self.kdtree.query(zip(xt, yt, zt))
        return mean_field.flatten()[indices].reshape(lonstarget.shape)


    def get_daily_clim_fields_interpolated_to(self, start_year=None, end_year=None,
                                              lons_target=None, lats_target=None):
        # Return 365 fields
        df = self.get_daily_climatology_fields(start_year=start_year, end_year=end_year)

        assert isinstance(df, Panel)

        lons1d, lats1d = lons_target.flatten(), lats_target.flatten()
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons1d, lats1d)

        dists, indices = self.kdtree.query(zip(xt, yt, zt))

        clim_fields = [
            df.loc[:, day, :].values.flatten()[indices].reshape(lons_target.shape) for day in df.major_axis
        ]
        clim_fields = np.asarray(clim_fields)
        clim_fields = np.ma.masked_where(np.isnan(clim_fields), clim_fields)
        return df.major_axis, clim_fields


def _reader(x):
    paths, day_of_month, nc_varname = x
    ds_list = [Dataset(path) for path in paths]
    mean_for_day = np.asarray([ds.variables[nc_varname][day_of_month - 1, :, :] for ds in ds_list]).mean(axis=0)
    [ds.close() for ds in ds_list]
    return mean_for_day


def demo_seasonal_mean():
    import matplotlib.pyplot as plt
    import crcm5.analyse_hdf.do_analysis_using_pytables as analysis
    # get target lons and lats for testing
    lon, lat, basemap = analysis.get_basemap_from_hdf(
        file_path="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf")

    x, y = basemap(lon, lat)

    am = AnuSplinManager()
    the_field = am.getMeanFieldForMonthsInterpolatedTo(start_year=1980, end_year=1988, lonstarget=lon, latstarget=lat)

    basemap.pcolormesh(x, y, the_field)
    basemap.colorbar()
    basemap.drawcoastlines()
    plt.show()


def demo():
    import application_properties

    application_properties.set_current_directory()
    am = AnuSplinManager()
    t0 = time.clock()
    df = am.get_daily_climatology_fields()
    print "Execution time: {0} seconds".format(time.clock() - t0)
    # x,t,y - the order of the axes



    print dir(df)
    df = df.fillna(value=np.nan)

    annual_mean = np.mean(df.values[:, :, :], axis=1)

    annual_mean = np.ma.masked_where(~(annual_mean == annual_mean), annual_mean)
    import matplotlib.pyplot as plt

    b = Basemap(resolution="l")

    x, y = b(am.lons2d, am.lats2d)

    plt.figure()
    b.pcolormesh(x, y, annual_mean.transpose())
    b.drawcoastlines()
    b.colorbar()

    monthly_means = df.groupby(lambda d: d.month).mean()

    for the_month in range(1, 13):
        plt.figure()
        plt.title("{0}".format(the_month))

        v = monthly_means.values[:, the_month - 1, :].transpose()
        v = np.ma.masked_invalid(v)
        b.pcolormesh(x, y, v)
        b.drawcoastlines()
        b.colorbar()

    plt.show()


def demo_interolate_daily_clim():
    import crcm5.analyse_hdf.do_analysis_using_pytables as analysis

    # get target lons and lats for testing
    lon, lat, basemap = analysis.get_basemap_from_hdf(
        file_path="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf")

    ans = AnuSplinManager()
    dates, fileds = ans.get_daily_clim_fields_interpolated_to(start_year=1980, end_year=1982,
                                                              lons_target=lon, lats_target=lat)
    import matplotlib.pyplot as plt

    plt.pcolormesh(fileds[5].transpose())

    print dates
    print fileds.mean(axis=1).mean(axis=1)


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    # demo()
    #demo_seasonal_mean()
    demo_interolate_daily_clim()
