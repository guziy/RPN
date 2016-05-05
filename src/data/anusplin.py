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
from data.base_data_manager import BaseDataManager
from util.geo import lat_lon

__author__ = 'huziy'
import pandas as pd

# precipitation data are in mm/day
# the temperature is in celsius

# Did not use MFDataset with this data, since not sure if it can handle the peculiar time in days


class AnuSplinManager(BaseDataManager):
    def __init__(self,
                 folder_path="/home/huziy/skynet1_rech3/anusplin_links",
                 variable="pcp",
                 file_name_preifx="ANUSPLIN_latlon_"):

        super().__init__()

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

        if os.path.isdir(os.path.realpath(folder_path)):
            self.folder_path = folder_path
        else:
            # Try a folder from skynet3
            print("{} does not exist".format(folder_path))
            self.folder_path = folder_path.replace("skynet1_rech3", "skynet3_rech1")
            print("Using {} instead".format(self.folder_path))

        self.fname_format = "{0}{1}".format(file_name_preifx, variable) + "_%Y_%m.nc"
        self._read_lon_lats()
        self.name = "ANUSPLIN"
        pass


    def get_seasonal_fields(self, start_year: int = -np.Inf, end_year: int = np.Inf, months: list = range(1, 13)) -> pd.Panel:

        months_str = "-".join([str(m) for m in months])
        cache_file = "anusplin_seasonal_means_{}_{}_{}_{}.cache.hdf".format(start_year, end_year, self.nc_varname, months_str)
        hdf_key = "seasonal_means_frame"
        if os.path.isfile(cache_file):
            store = pd.HDFStore(cache_file)
            df = store.get(hdf_key)
            store.close()
            return df


        data = []
        year_axis = []

        for y in range(start_year, end_year + 1):
            print("reading fields for {} .... ".format(y))



            files = [os.path.join(self.folder_path, datetime(y, m, 1).strftime(self.fname_format)) if m > 2 else os.path.join(self.folder_path, datetime(y + 1, m, 1).strftime(self.fname_format)) for m in months if not (y == end_year and m in [12, 1, 2])]


            if y == end_year:
                if set(months) == {1, 2, 12}:
                    assert len(files) == 0


            mean_year = None
            counter_year = 0
            for fp in files:
                ds = Dataset(fp)
                print(ds)
                v = ds.variables[self.nc_varname][:]

                print(v.shape[0])
                if mean_year is None:
                    mean_year = v.sum(axis=0)
                    counter_year = v.shape[0]
                else:
                    mean_year += v.sum(axis=0)
                    counter_year += v.shape[0]

                ds.close()

            if mean_year is None:
                continue

            year_axis.append(y)
            data.append(mean_year / counter_year)



        data = np.array(data)
        panel = pd.Panel(data=data.transpose((1, 0, 2)), items=range(data.shape[1]), major_axis=year_axis, minor_axis=range(data.shape[2]))

        panel.to_hdf(cache_file, hdf_key)

        return panel


    def _read_lon_lats(self):
        """
        Read the lons and lats, 2d from the first file
        """
        the_date = datetime(1980, 1, 1)
        fpath = os.path.join(self.folder_path, the_date.strftime(self.fname_format))

        # Check if file exists before trying to read it
        if not os.path.isfile(fpath):
            raise IOError("No such file: {}".format(fpath))

        ds = Dataset(fpath)
        self.lons2d = ds.variables["lon"][:].transpose()
        self.lats2d = ds.variables["lat"][:].transpose()

        x, y, z = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
        self.kdtree = cKDTree(list(zip(x, y, z)))

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

                month_days = ds.variables["time"][:] if month != 2 else list(range(1, 29))
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
        # np.asarray([ds.variables[self.nc_varname][day - 1, :, :] for ds in ds_list]).mean(axis=0)
        #    for day, ds_list in zip(day_list, ds_lists)
        # ]

        daily_clim_fields = pool.map(_reader, list(zip(path_lists, day_of_month_list, varname_list)))
        daily_clim_fields = np.asarray(daily_clim_fields)
        daily_clim_fields = np.ma.masked_where(np.isnan(daily_clim_fields), daily_clim_fields)

        ni, nj = daily_clim_fields[0].shape
        panel = pd.Panel(data=daily_clim_fields.transpose((1, 0, 2)), major_axis=times, minor_axis=list(range(nj)),
                         items=list(range(ni)))

        store = pd.HDFStore(cache_file)
        store.put("dailyClimFrame", panel)
        store.close()

        return panel


    def getMeanFieldForMonths(self, months=None, start_year=1979, end_year=1988):
        if months is None:
            months = list(range(1, 13))

        df = self.get_daily_climatology_fields(start_year=start_year, end_year=end_year)

        assert isinstance(df, Panel)

        df_monthly = df.select(lambda d: d.month in months, axis=1)

        df_mean = df_monthly.apply(np.mean)

        assert isinstance(df_mean, pd.DataFrame)
        print(df_mean.values.shape)

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

        dists, indices = self.kdtree.query(list(zip(xt, yt, zt)))
        return mean_field.flatten()[indices].reshape(lonstarget.shape)



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
    print("Execution time: {0} seconds".format(time.clock() - t0))
    # x,t,y - the order of the axes



    print(dir(df))
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


def _get_topography():
    path = "/skynet3_rech1/huziy/geofields_interflow_exp/geophys_Quebec_0.1deg_260x260_with_dd_v6_with_ITFS"
    from rpn.rpn import RPN
    r = RPN(path=path)
    data = r.get_first_record_for_name_and_level("ME", level=0)
    r.close()
    return data


def demo_interolate_daily_clim():
    import crcm5.analyse_hdf.do_analysis_using_pytables as analysis

    model_data_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r.hdf5"
    start_year = 1980
    end_year = 2010

    vmin = -30
    vmax = 30
    vname = "TT_max"
    coef_mod = 1.0e3 * 24 * 3600 if vname == "PR" else 1.0

    # get target lons and lats for testing
    lon, lat, basemap = analysis.get_basemap_from_hdf(
        file_path=model_data_path)

    ans = AnuSplinManager(variable="stmx")
    dates, fields = ans.get_daily_clim_fields_interpolated_to(start_year=start_year,
                                                              end_year=end_year,
                                                              lons_target=lon, lats_target=lat)
    import matplotlib.pyplot as plt

    x, y = basemap(lon, lat)

    margin = 20
    topo = _get_topography()[margin:-margin, margin:-margin]

    # Plot obs data
    plt.figure()
    mean_obs = np.ma.array([fields[i] for i, d in enumerate(dates) if d.month in range(1, 13)]).mean(axis=0)
    im = basemap.pcolormesh(x, y, mean_obs, vmin=vmin, vmax=vmax)
    basemap.colorbar(im)
    basemap.drawcoastlines()
    plt.title("Anusplin")
    print("Obs stdev = {}".format(mean_obs[~mean_obs.mask].std()))

    print("Obs correlations: ", np.corrcoef(mean_obs[~mean_obs.mask], topo[~mean_obs.mask]))

    # Plot model data
    plt.figure()
    dates, fields = analysis.get_daily_climatology(path_to_hdf_file=model_data_path, var_name=vname,
                                                   level=0,
                                                   start_year=start_year, end_year=end_year)

    mean_mod = np.array([fields[i] for i, d in enumerate(dates) if d.month in range(1, 13)]).mean(axis=0) * coef_mod
    im = basemap.pcolormesh(x, y, mean_mod, vmin=vmin, vmax=vmax)
    basemap.colorbar(im)
    basemap.drawcoastlines()
    plt.title("Model")

    print("Model correlations: ", np.corrcoef(mean_mod[~mean_obs.mask], topo[~mean_obs.mask]))
    print("Model stdev = {}".format(mean_mod[~mean_obs.mask].std()))

    plt.show()


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    # demo()
    # demo_seasonal_mean()
    demo_interolate_daily_clim()
