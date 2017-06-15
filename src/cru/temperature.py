import calendar
from datetime import timedelta, datetime
import itertools
import collections
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pandas
from scipy.spatial.ckdtree import cKDTree
from scipy.spatial.kdtree import KDTree
import application_properties
from crcm5.model_data import Crcm5ModelDataManager
from crcm5.model_point import ModelPoint
from data.timeseries import TimeSeries

from util.geo import lat_lon
from util.seasons_info import MonthPeriod

__author__ = 'huziy'

import numpy as np
from netCDF4 import Dataset, num2date, date2num
import matplotlib.pyplot as plt
from collections import OrderedDict


class CRUDataManager:
    def __init__(self, path="/RECH/skynet1_rech3/huziy/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.tmp.dat.nc",
                 var_name="tmp", lazy=False):

        self.times = None
        self.var_data = None

        self.times_var = None
        self.kdtree = None
        self.times_num = None
        self.lons2d, self.lats2d = None, None

        self.lazy = lazy
        self.var_name = var_name

        with Dataset(path) as ds:
            self._init_fields(ds)

        # Cannot go into with, since it needs to be open
        self.nc_dataset = Dataset(path)
        self.nc_vars = ds.variables


    def _init_fields(self, nc_dataset):
        nc_vars = nc_dataset.variables
        lons = nc_vars["lon"][:]
        lats = nc_vars["lat"][:]

        lats2d, lons2d = np.meshgrid(lats, lons)

        self.lons2d, self.lats2d = lons2d, lats2d

        self.times_var = nc_vars["time"]
        self.times_num = nc_vars["time"][:]

        if hasattr(self.times_var, "calendar"):
            self.times = num2date(self.times_num, self.times_var.units, self.times_var.calendar)
        else:
            self.times = num2date(self.times_num, self.times_var.units)

        if not self.lazy:
            self.var_data = np.transpose(nc_vars[self.var_name][:], axes=[0, 2, 1])

        x_in, y_in, z_in = lat_lon.lon_lat_to_cartesian(self.lons2d.flatten(), self.lats2d.flatten())
        self.kdtree = cKDTree(list(zip(x_in, y_in, z_in)))








    def get_seasonal_means_with_ttest_stats_interp_to(self, lons2d=None, lats2d=None,
                                                      season_to_monthperiod=None, start_year=None, end_year=None):

        #TODO: implement
        pass





    def get_seasonal_means_with_ttest_stats(self, season_to_monthperiod=None, start_year=None, end_year=None):
        """
        Note: the periods of different seasons should not overlap.


        precip are converted to mm/day before the mean and std calculations

        :param season_to_monthperiod: 
        :param start_year: 
        :param end_year:
        :return dict(season: [mean, std, nobs])
        """

        nt, nx, ny = self.var_data.shape
        panel = pandas.Panel(data=self.var_data, items=self.times, major_axis=list(range(nx)), minor_axis=list(range(ny)))
        panel = panel.select(lambda d: start_year <= d.year <= end_year)

        # Calculate monthly means, convert precip to mm/day
        if self.var_name.lower() in ["pre"]:
            monthly_panel = panel.groupby(lambda d: (d.year, d.month), axis="items").sum()

            monthly_panel = pandas.Panel(data=monthly_panel.values / monthly_panel.items.map(lambda ym: calendar.monthrange(*ym)[1])[:, np.newaxis, np.newaxis],
                                                         items=monthly_panel.items,
                                                         minor_axis=monthly_panel.minor_axis,
                                                         major_axis=monthly_panel.major_axis)

        else:
            monthly_panel = panel.groupby(lambda d: (d.year, d.month), axis="items").mean()


        season_to_res = {}

        for season, month_period in season_to_monthperiod.items():
            assert isinstance(month_period, MonthPeriod)

            print("{} ------- (months: {}) ".format(season, month_period.months))

            ym_to_period = month_period.get_year_month_to_period_map(start_year=start_year, end_year=end_year)
            print(ym_to_period)

            # select data for the seasons of interest
            monthly_panel_tmp = monthly_panel.select(lambda ym: ym[1] and (ym in ym_to_period) in month_period.months)

            days_per_month = monthly_panel_tmp.items.map(lambda ym: calendar.monthrange(*ym)[1])


            monthly_panel_tmp = pandas.Panel(data=monthly_panel_tmp.values * days_per_month[:, np.newaxis, np.newaxis],
                                             major_axis=monthly_panel_tmp.major_axis,
                                             minor_axis=monthly_panel_tmp.minor_axis,
                                             items=monthly_panel_tmp.items)


            seasonal_groups = monthly_panel_tmp.groupby(lambda ym: (ym_to_period[ym].start,  ym_to_period[ym].end), axis="items")

            nobs = len(seasonal_groups)


            seasonal_means = []
            days_per_season = []


            for kv, gv in seasonal_groups:
                print(kv, "---->", gv)

                # calculate seasonal mean for each year
                ndays = (kv[1] - kv[0]).days
                seas_mean = gv.values.sum(axis=0) / ndays

                seasonal_means.append(seas_mean)
                days_per_season.append(ndays)



            seasonal_means = np.array(seasonal_means)
            days_per_season = np.array(days_per_season)

            # calculate climatological mean
            clim_mean = (seasonal_means * days_per_season[:, np.newaxis, np.newaxis]).sum(axis=0) / days_per_season.sum()



            # calculate interannual std
            clim_std = (((seasonal_means - clim_mean) ** 2 * days_per_season[:, np.newaxis, np.newaxis]).sum(axis=0) / days_per_season.sum()) ** 0.5



            spatial_mask = clim_mean > 1e10

            clim_mean = np.ma.masked_where(spatial_mask, clim_mean)
            clim_std = np.ma.masked_where(spatial_mask, clim_std)


            season_to_res[season] = [clim_mean, clim_std, nobs]

        return season_to_res




    def get_seasonal_means(self, season_name_to_months=None, start_year=None, end_year=None):
        if season_name_to_months is None:
            season_name_to_months = OrderedDict([
                ("Winter", (1, 2, 12)),
                ("Spring", list(range(3, 6))),
                ("Summer", list(range(6, 9))),
                ("Fall", list(range(9, 12)))])

        season_name_to_coef = {}
        for sname, months in season_name_to_months.items():
            season_name_to_coef[sname] = 1

            if self.var_name.lower() in ["pre", "precip"]:
                days = sum([calendar.monthrange(y, m)[1] for m in months for y in range(start_year, end_year + 1)])
                season_name_to_coef[sname] = 1.0 / float(days)

        month_to_season = collections.defaultdict(lambda: "no_season")
        for sname, mlist in season_name_to_months.items():
            for m in mlist:
                month_to_season[m] = sname

        if self.var_data is None:
            self.var_data = self.nc_dataset.variables[self.var_name][:]
            if self.var_name.lower() not in ["swe"]:
                self.var_data = np.transpose(self.var_data, axes=[0, 2, 1])

        nt, nx, ny = self.var_data.shape
        panel = pandas.Panel(data=self.var_data, items=self.times, major_axis=list(range(nx)),
                             minor_axis=list(range(ny)))
        panel = panel.select(lambda d: start_year <= d.year <= end_year)

        if self.var_name in ["pre", "precip"]:
            panel_seasonal = panel.groupby(lambda d: month_to_season[d.month], axis="items").sum()
        else:
            panel_seasonal = panel.groupby(lambda d: month_to_season[d.month], axis="items").mean()

        season_to_mean = OrderedDict()
        for sname, _ in season_name_to_months.items():
            season_to_mean[sname] = panel_seasonal[sname].values * season_name_to_coef[sname]
            if hasattr(self.var_data[0], "mask"):
                season_to_mean[sname] = np.ma.masked_where(self.var_data[0].mask, season_to_mean[sname])

        return season_to_mean


    def get_mean(self, start_year, end_year, months=None):
        """
        returns the mean for the period [start_year, end_year], over the months
        :type months: list
        months = list of month numbers over which the averaging is done
        """

        if months is None:
            months = list(range(1, 13))

        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year + 1, 1, 1)

        start_date_num = date2num(start_date, self.times_var.units)
        end_date_num = date2num(end_date, self.times_var.units)

        sel_query = (self.times_num >= start_date_num) & (self.times_num < end_date_num)
        sel_dates = self.times_num[sel_query]
        sel_data = np.transpose(self.nc_vars[self.var_name][sel_query, :, :], axes=[0, 2, 1])

        sel_dates = num2date(sel_dates, self.times_var.units)

        ind_vector = np.where([(x.month in months) for x in sel_dates])[0]
        return np.mean(sel_data[ind_vector, :, :], axis=0)


    def get_daily_climatology_dataframe(self, start_year, end_year, stamp_year=2001):
        """
        returns a pandas dataframe (365, nx, ny) with daily climatological means
        """
        nt, nx, ny = self.var_data.shape
        data_panel = pandas.Panel(data=self.var_data, items=self.times, major_axis=list(range(nx)),
                                  minor_axis=list(range(ny)))
        data_panel = data_panel.select(
            lambda d: (start_year <= d.year <= end_year) and not (d.day == 29 and d.month == 2))

        data_panel = data_panel.groupby(lambda d: datetime(stamp_year, d.month, d.day), axis="items").mean()
        assert isinstance(data_panel, pandas.Panel)
        data_panel = data_panel.sort_index()
        print(data_panel.values.shape)
        return data_panel


    def get_daily_climatology(self, start_year, end_year, stamp_year=2001):
        """
        returns a numpy array of shape (365, nx, ny) with daily climatological means
        """
        return self.get_daily_climatology_dataframe(**locals()).values


    def interpolate_daily_climatology_to(self, clim_data, lons2d_target=None, lats2d_target=None):
        # expects clim_data to have the following shape (365, nx, ny)
        #        lons2d_target: (nx, ny)
        #        lats2d_target: (nx, ny)


        x, y, z = lat_lon.lon_lat_to_cartesian(lons2d_target.flatten(), lats2d_target.flatten())

        nt = clim_data.shape[0]
        data_help = np.reshape(clim_data, (nt, -1))

        dists, inds = self.kdtree.query(list(zip(x, y, z)))

        return data_help[:, inds].reshape((nt,) + lons2d_target.shape)

        pass


    def get_thawing_index_from_climatology(self, daily_temps_clim, t0=0.0):

        nt, nx, ny = daily_temps_clim.shape
        result = np.zeros((nx, ny))

        for t in range(nt):
            tfield = daily_temps_clim[t, :, :]
            result += tfield * np.array(tfield >= t0).astype(int)
        return result


    def create_monthly_means_file(self, start_year, end_year):
        fname = "{0}_monthly_means.nc".format(self.var_name)
        year_range = list(range(start_year, end_year + 1))
        dsm = Dataset(fname, "w", format="NETCDF3_CLASSIC")
        dsm.createDimension('year', len(year_range))
        dsm.createDimension("month", 12)
        dsm.createDimension('lon', self.lons2d.shape[0])
        dsm.createDimension('lat', self.lons2d.shape[1])

        lonVariable = dsm.createVariable('longitude', 'f4', ('lon', 'lat'))
        latVariable = dsm.createVariable('latitude', 'f4', ('lon', 'lat'))
        yearVariable = dsm.createVariable("year", "i4", ("year",))

        variable = dsm.createVariable(self.var_name, "f4", ('year', "month", 'lon', 'lat'))
        for i, the_year in enumerate(year_range):
            print(the_year)
            for j, the_month in enumerate(range(1, 13)):
                variable[i, j, :, :] = self.get_mean(the_year, the_year, months=[the_month])

        lonVariable[:] = self.lons2d
        latVariable[:] = self.lats2d
        yearVariable[:] = np.array(year_range)
        dsm.close()

        pass


    def _interp_and_sum(self, data1d, mults_1d, x, y, z, nneighbors=1):
        data_interp = self.interpolate_data_to_cartesian(data1d, x, y, z, nneighbours=nneighbors)
        return np.sum(mults_1d * data_interp)

    def get_monthly_timeseries_using_mask(self, mask, lons2d_target, lats2d_target, multipliers_2d, start_date=None,
                                          end_date=None):
        """
        multipliers_2d used to multiply the values when aggregating into a single timeseries
        sum(mi * vi) - in space
        """

        bool_vect = np.array([start_date <= t <= end_date for t in self.times])

        new_times = list(filter(lambda t: start_date <= t <= end_date, self.times))
        new_vals = self.var_data[bool_vect, :, :]
        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d_target.flatten(), lats2d_target.flatten())

        print(len(new_times))
        flat_mask = mask.flatten()
        x_out = x_out[flat_mask == 1]
        y_out = y_out[flat_mask == 1]
        z_out = z_out[flat_mask == 1]
        mults = multipliers_2d.flatten()[flat_mask == 1]

        data_interp = [self._interp_and_sum(new_vals[t, :, :].flatten(), mults, x_out, y_out, z_out) for t in
                       range(len(new_times))]

        print("Interpolated data", data_interp)

        print("Interpolated all")
        return TimeSeries(time=new_times, data=data_interp).get_ts_of_monthly_means()


    def get_mean_upstream_timeseries_monthly(self, model_point, data_manager):
        """
        get mean swe upstream of the model_point

        year range for selection is in model_point.continuous_data_years() ..
        """
        assert isinstance(model_point, ModelPoint)
        assert isinstance(data_manager, Crcm5ModelDataManager)



        # create the mask of points over which the averaging is going to be done
        lons_targ = data_manager.lons2D[model_point.flow_in_mask == 1]
        lats_targ = data_manager.lats2D[model_point.flow_in_mask == 1]

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_targ, lats_targ)

        nxs, nys = self.lons2d.shape
        i_source, j_source = list(range(nxs)), list(range(nys))

        j_source, i_source = np.meshgrid(j_source, i_source)

        i_source = i_source.flatten()
        j_source = j_source.flatten()

        dists, inds = self.kdtree.query(list(zip(xt, yt, zt)), k=1)
        ixsel = i_source[inds]
        jysel = j_source[inds]

        print("Calculating spatial mean")
        #calculate spatial mean
        #calculate spatial mean
        if self.lazy:
            theVar = self.nc_vars[self.var_name]

            data_series = []
            for i, j in zip(ixsel, jysel):
                data_series.append(theVar[:, j, i])

            data_series = np.mean(data_series, axis=0)
        else:
            data_series = np.mean(self.var_data[:, ixsel, jysel], axis=1)

        print("Finished calculating spatial mean")

        #calculate daily climatology
        df = pandas.DataFrame(data=data_series, index=self.times, columns=["values"])

        df["year"] = df.index.map(lambda d: d.year)

        df = df[df["year"].isin(model_point.continuous_data_years)]
        monthly_clim = df.groupby(by=lambda d: d.month).mean()

        month_dates = [datetime(1985, m, 15) for m in range(1, 13)]
        vals = [monthly_clim.ix[d.month, "values"] for d in month_dates]

        return pandas.TimeSeries(data=vals, index=month_dates)


    def get_mean_upstream_timeseries_daily(self, model_point, dm, stamp_dates=None):
        """
        get mean swe upstream of the model_point
        """
        assert isinstance(model_point, ModelPoint)

        assert isinstance(dm, Crcm5ModelDataManager)



        # create the mask of points over which the averaging is going to be done
        lons_targ = dm.lons2D[model_point.flow_in_mask == 1]
        lats_targ = dm.lats2D[model_point.flow_in_mask == 1]

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_targ, lats_targ)

        nxs, nys = self.lons2d.shape
        i_source, j_source = list(range(nxs)), list(range(nys))

        j_source, i_source = np.meshgrid(j_source, i_source)

        i_source = i_source.flatten()
        j_source = j_source.flatten()

        dists, inds = self.kdtree.query(list(zip(xt, yt, zt)), k=1)
        ixsel = i_source[inds]
        jysel = j_source[inds]

        df_empty = pandas.DataFrame(index=self.times)
        df_empty["year"] = df_empty.index.map(lambda d: d.year)

        # calculate spatial mean
        sel_date_indices = np.where(df_empty["year"].isin(model_point.continuous_data_years))[0]
        if self.lazy:
            the_var = self.nc_vars[self.var_name]
            data_series = np.mean([the_var[sel_date_indices, j, i] for i, j in zip(ixsel, jysel)], axis=0)
        else:
            data_series = np.mean(self.var_data[:, ixsel, jysel], axis=1)


        # calculate daily climatology
        df = pandas.DataFrame(data=data_series, index=self.times, columns=["values"])

        df["year"] = df.index.map(lambda d: d.year)
        df = df[df["year"].isin(model_point.continuous_data_years)]
        daily_clim = df.groupby(by=lambda d: (d.month, d.day)).mean()

        vals = [daily_clim.ix[(d.month, d.day), "values"] for d in stamp_dates]
        return pandas.TimeSeries(data=vals, index=stamp_dates)


    def get_daily_timeseries_using_mask(self, mask, lons2d_target, lats2d_target, multipliers_2d, start_date=None,
                                        end_date=None):
        """
        multipliers_2d used to multiply the values when aggregating into a single timeseries
        sum(mi * vi) - in space
        """

        bool_vect = np.array([start_date <= t <= end_date for t in self.times])

        new_times = list(filter(lambda t: start_date <= t <= end_date, self.times))
        new_vals = self.var_data[bool_vect, :, :]
        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d_target.flatten(), lats2d_target.flatten())

        print(len(new_times))

        flat_mask = mask.flatten()
        x_out = x_out[flat_mask == 1]
        y_out = y_out[flat_mask == 1]
        z_out = z_out[flat_mask == 1]
        mults = multipliers_2d.flatten()[flat_mask == 1]
        data_interp = [self._interp_and_sum(new_vals[t, :, :].flatten(), flat_mask, x_out, y_out, z_out) for t in
                       range(len(new_times))]

        print("Interpolated all")
        return TimeSeries(time=new_times, data=data_interp).get_ts_of_daily_means()


    def interpolate_data_to_cartesian(self, data_in_flat, x, y, z, nneighbours=4):
        """
        len(data_in_flat) , len(x) == len(y) == len(z) == len(data_out_flat) - all 1D
        """
        print("start query")
        dst, ind = self.kdtree.query(list(zip(x, y, z)), k=nneighbours)
        print("end query")

        inverse_square = 1.0 / dst ** 2
        if len(dst.shape) > 1:
            norm = np.sum(inverse_square, axis=1)
            norm = np.array([norm] * dst.shape[1]).transpose()
            coefs = inverse_square / norm

            data_out_flat = np.sum(coefs * data_in_flat[ind], axis=1)
        elif len(dst.shape) == 1:
            data_out_flat = data_in_flat[ind]
        else:
            raise Exception("Could not find neighbor points")
        return data_out_flat


    def interpolate_data_to(self, data_in, lons2d, lats2d, nneighbours=4):
        """
        Interpolates data_in to the grid defined by (lons2d, lats2d)
        assuming that the data_in field is on the initial CRU grid

        interpolate using 4 nearest neighbors and inverse of squared distance
        """

        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
        dst, ind = self.kdtree.query(list(zip(x_out, y_out, z_out)), k=nneighbours)

        data_in_flat = data_in.flatten()

        inverse_square = 1.0 / dst ** 2
        if len(dst.shape) > 1:
            norm = np.sum(inverse_square, axis=1)
            norm = np.array([norm] * dst.shape[1]).transpose()
            coefs = inverse_square / norm

            data_out_flat = np.sum(coefs * data_in_flat[ind], axis=1)
        elif len(dst.shape) == 1:
            data_out_flat = data_in_flat[ind]
        else:
            raise Exception("Could not find neighbor points")
        return np.reshape(data_out_flat, lons2d.shape)


def main():
    from permafrost import draw_regions

    dm = CRUDataManager()

    b, lons2d, lats2d = draw_regions.get_basemap_and_coords()

    x, y = b(dm.lons2d, dm.lats2d)

    fig = plt.figure()

    gs = gridspec.GridSpec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    data = dm.get_mean(1981, 2009, months=[6, 7, 8])
    img = b.contourf(x, y, data.copy(), ax=ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img, cax=cax)
    b.drawcoastlines(ax=ax)
    ax.set_title("CRU (not interp.), \n JJA period: {0} - {1}".format(1981, 2009))

    ax = fig.add_subplot(gs[0, 1])
    data_projected = dm.interpolate_data_to(data, lons2d, lats2d)
    x, y = b(lons2d, lats2d)
    img = b.contourf(x, y, data_projected, ax=ax, levels=img.levels)

    # add pretty colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = fig.colorbar(img, cax=cax)

    b.drawcoastlines(ax=ax)
    ax.set_title("CRU ( interp.), \n JJA period: {0} - {1}".format(1981, 2009))

    plt.show()
    plt.savefig("t_cru_jja.png")

    pass


def create_monthly_means():
    # tmp
    #dm = CRUDataManager()
    #dm.create_monthly_means_file(1901, 2009)

    #pre
    dm = CRUDataManager(path="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc", var_name="pre")
    dm.create_monthly_means_file(1901, 2009)


def plot_thawing_index():
    dm = CRUDataManager()
    clim = dm.get_daily_climatology(1981, 2010)
    thi = dm.get_thawing_index_from_climatology(clim)

    plt.pcolormesh(thi.transpose())
    plt.colorbar()
    plt.show()



def test_get_seasonal_means_with_ttest_stats():

    manager = CRUDataManager(path="/HOME/data/Validation/CRU_TS_3.1/Original_files_gzipped/cru_ts_3_10.1901.2009.tmp.dat.nc",
                             var_name="tmp")


    season_to_month_period = OrderedDict([
        ("DJF", MonthPeriod(12, 3))
    ])

    res = manager.get_seasonal_means_with_ttest_stats(
        season_to_monthperiod=season_to_month_period, start_year=1980, end_year=1982)

    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    # plot_thawing_index()
    # create_monthly_means()
    # main()

    test_get_seasonal_means_with_ttest_stats()

    print("Hello world")
  
