import calendar
from collections import defaultdict

import biggus
import dask
import dask.array as da
import numpy as np
from dask.local import get_sync
from netCDF4 import Dataset, MFDataset, num2date
from rpn.domains import lat_lon
from scipy.spatial import KDTree

from collections import OrderedDict

from util.seasons_info import MonthPeriod
import pandas as pd

dask.set_options(get=get_sync)  # use single-threaded scheduler by default


class HighResDataManager(object):
    def __init__(self, path="", vname="", characteristic_scale_deg=0.01, chunks=(5, 500, 500)):

        self.chunks = chunks

        try:
            self.__ds = Dataset(path)
            self.data = da.from_array(Dataset(path).variables[vname], self.chunks, lock=True)
        except OSError as err:

            import glob

            if isinstance(path, str):
                path_list = glob.glob(path)
            else:
                path_list = path

            path_list = sorted(path_list)

            self.data = [da.from_array(Dataset(p).variables[vname], self.chunks, lock=True) for p in path_list]
            self.data = da.concatenate(self.data)


            self.__ds = MFDataset(path_list)



        self.missing_value = None

        if hasattr(self.__ds.variables[vname], "missing_value"):
            self.missing_value = self.__ds.variables[vname].missing_value



        self.vname = vname

        #
        # self.data = biggus.OrthoArrayAdapter(self.ds.variables[vname])


        self.lons = None
        self.lats = None
        self.time = None

        self.time_to_index = None

        self.characteristic_scale_deg = characteristic_scale_deg



        self.__read_coordinates_and_time()
        self.__ds.close()


    def get_data_aggregated_in_space(self, chunk_size):
        return self.data.rechunk(chunks=chunk_size).map_blocks()



    def get_annual_max_with_ttest_stats_lazy(self, data, start_year=-np.Inf, end_year=np.Inf):
        """
        Get the maximum for each year, calculate clim_mean and standard deviation, to be able to use the in ttest
        :param data:
        :param start_year:
        :param end_year:
        :return (mean of ann max, std of ann max, nyears), mask
        """

        data_sel, time_sel = self.__sel_period(start_year=start_year, end_year=end_year, arr=data)

        data_sel = data_sel.rechunk((len(time_sel),) + self.chunks.shape[1:])


        mask = np.abs(data_sel[0, :, :] - self.missing_value) < 1.0e-6

        def annual_max(block):
            tmp = block.reshape((len(time_sel), -1))

            df = pd.DataFrame(index=time_sel, data=tmp)

            return df.groupby(lambda d: d.year, sort=True).max().values.reshape((-1, ) + block.shape[1:])




        ann_max_arr = data_sel.map_blocks(annual_max)

        # get climatology and standard deviations
        ann_max_mean_clim = ann_max_arr.mean(axis=0)
        ann_max_std = ann_max_arr.std(axis=0)

        return ann_max_mean_clim, ann_max_std, ann_max_arr.shape[0], mask




    def get_daily_percenile_fields_interpolated_to(self, lons_target, lats_target, start_year=-np.Inf, end_year=np.Inf, percentile=0.5,
                                                   rolling_mean_window_days=None):
        target_scale_deg = (lons_target[1, 1] - lons_target[0, 0] + lats_target[1, 1] - lats_target[0, 0]) / 2.0

        coarsening = int(target_scale_deg / self.characteristic_scale_deg + 0.5)
        print("source_scale: {}\ntarget_scale: {}\ncoarsening coefficient: {}".format(self.characteristic_scale_deg,
                                                                                      target_scale_deg, coarsening))

        def coarsening_func(x, axis=None):
            _mask = np.less(np.abs(x - self.missing_value), 1.0e-6)

            if np.all(_mask):
                return self.missing_value * np.ma.ones(_mask.shape).mean(axis=axis)

            y = np.ma.masked_where(_mask, x)

            return y.mean(axis=axis)

        # aggregate the data
        trim_excess = True
        data = da.coarsen(coarsening_func, self.data, axes={1: coarsening, 2: coarsening}, trim_excess=trim_excess)
        lons_s = da.coarsen(np.mean, da.from_array(self.lons, self.chunks[1:]), axes={0: coarsening, 1: coarsening},
                            trim_excess=trim_excess).compute()
        lats_s = da.coarsen(np.mean, da.from_array(self.lats, self.chunks[1:]), axes={0: coarsening, 1: coarsening},
                            trim_excess=trim_excess).compute()

        source_grid = list(zip(*lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())))
        print(np.shape(source_grid))
        ktree = KDTree(source_grid)

        dists, inds = ktree.query(
            list(zip(*lat_lon.lon_lat_to_cartesian(lons_target.flatten(), lats_target.flatten()))))


        perc_daily, mask = self.get_daily_percenile_fields_lazy(data, start_year=start_year, end_year=end_year,
                                                          percentile=percentile, rolling_mean_window_days=rolling_mean_window_days)


        print("perc_daily.shape=", perc_daily.shape)

        # do the interpolation for each day
        perc_daily_interpolated = []
        for perc_field in perc_daily:
            print(perc_field.shape)
            field = np.ma.masked_where(mask, perc_field.compute()).flatten()[inds].reshape(lons_target.shape)
            perc_daily_interpolated.append(field)

        return np.array(perc_daily_interpolated)


    # @profile
    def get_daily_percenile_fields_lazy(self, data, start_year=-np.Inf, end_year=np.Inf, percentile=0.5, rolling_mean_window_days=None):

        """
        calculate the percentile for each day of year for the specified period
        :param rolling_mean_window_days: if None[default] the rolling mean is not applied, if 1 or N - the rolling mean of 1 or N days is applied before computing the percentile
        :param percentile: ranges from 0 to 1.0
        :param data: (time, lon, lat) dask array
        :param start_year:
        :param end_year:
        :return : 365 mean fields (1 for each day of year) of <var>percentile</var> percentile, and the mask

        """
        assert isinstance(data, da.Array)



        msg = "The first dimension of data, should be time, but data.shape[0]={} and len(self.time)={}".format(data.shape[0], len(self.time))
        assert data.shape[0] == len(self.time), msg

        # mask the resulting fields
        epsilon = 1.0e-5
        mask = np.less_equal(np.abs(data[0, :, :] - self.missing_value), epsilon)


        data_sel, time_sel = data, self.time

        assert np.all(np.equal(sorted(time_sel), time_sel)), "Time vector does not appear to be sorted"



        print("start rechunking")
        data_sel = data_sel.rechunk((len(time_sel),) + data_sel.chunks[1:])
        print("finish rechunking")


        perc = data_sel.map_blocks(percentile_calculator, time_sel, dtype=np.float32,
                                   rolling_mean_window_days=rolling_mean_window_days, percentile=percentile,
                                   start_year=start_year, end_year=end_year)

        return perc, mask





    def get_seasonal_means_with_ttest_stats_interpolated_to(self, lons_target, lats_target,
                                                            season_to_monthperiod=None, start_year=-np.Inf, end_year=np.Inf,
                                                            convert_monthly_accumulators_to_daily=False):


        """

        :param lons_target, lats_target: 2d arrays of target longitudes and latitudes
        :param season_to_monthperiod:
        :param start_year:
        :param end_year:
        :param convert_monthly_accumulators_to_daily: if true converts monthly accumulators to daily,
        :return dict(season: [mean, std, nobs])


        # coarsen the data and coordinates to the target scale and interpolate using nearest neighbours
        """


        target_scale_deg = (lons_target[1, 1] - lons_target[0, 0] + lats_target[1, 1] - lats_target[0, 0]) / 2.0


        coarsening = int(target_scale_deg / self.characteristic_scale_deg + 0.5)
        print("source_scale: {}\ntarget_scale: {}\ncoarsening coefficient: {}".format(self.characteristic_scale_deg, target_scale_deg, coarsening))

        def coarsening_func(x, axis=None):
            _mask = np.less(np.abs(x - self.missing_value), 1.0e-6)


            if np.all(_mask):
                return self.missing_value * np.ma.ones(_mask.shape).mean(axis=axis)

            y = np.ma.masked_where(_mask, x)

            return y.mean(axis=axis)


        # aggregate the data
        trim_excess = True
        data = da.coarsen(coarsening_func, self.data, axes={1: coarsening, 2: coarsening}, trim_excess=trim_excess)
        lons_s = da.coarsen(np.mean, da.from_array(self.lons, self.chunks[1:]), axes={0: coarsening, 1: coarsening}, trim_excess=trim_excess).compute()
        lats_s = da.coarsen(np.mean, da.from_array(self.lats, self.chunks[1:]), axes={0: coarsening, 1: coarsening}, trim_excess=trim_excess).compute()





        source_grid = list(zip(*lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())))
        print(np.shape(source_grid))
        ktree = KDTree(source_grid)

        dists, inds = ktree.query(list(zip(*lat_lon.lon_lat_to_cartesian(lons_target.flatten(), lats_target.flatten()))))



        print("data.shape = ", data.shape)
        result, mask = self.__get_seasonal_means_with_ttest_stats_dask_lazy(data, season_to_monthperiod=season_to_monthperiod,
                                                             start_year=start_year, end_year=end_year,
                                                             convert_monthly_accumulators_to_daily=convert_monthly_accumulators_to_daily)


        # invoke the computations and interpolate the result
        for season in result:
            print("Computing for {}".format(season))
            for i in range(len(result[season]) - 1):

                result[season][i] = np.ma.masked_where(mask, result[season][i].compute()).flatten()[inds].reshape(lons_target.shape)


        return result










    def __read_coordinates_and_time(self):
        for nc_vname, nc_var in self.__ds.variables.items():
            vname_lc = nc_vname.lower()

            if "lon" in vname_lc:
                self.lons = nc_var[:]
            elif "lat" in vname_lc:
                self.lats = nc_var[:]

            elif "time" in vname_lc and "bnds" not in vname_lc:
                if not hasattr(nc_var, "calendar"):
                    self.time = num2date(nc_var[:], nc_var.units)
                else:
                    print("Found the calendar attribute, using calendar={}".format(nc_var.calendar))
                    self.time = num2date(nc_var[:], nc_var.units, calendar=nc_var.calendar)




        if self.lons.ndim == 1:
            self.lats, self.lons = np.meshgrid(self.lats, self.lons)


        if self.lons.shape != self.data.shape[1:]:
            self.data = self.data.transpose(axis=[0, 2, 1])






    def get_seasonal_means_with_ttest_stats(self, season_to_monthperiod=None, start_year=None, end_year=None,
                                            convert_monthly_accumulators_to_daily=False):


        """

        :param season_to_monthperiod:
        :param start_year:
        :param end_year:
        :param convert_monthly_accumulators_to_daily: if true converts monthly accumulators to daily,
        :return dict(season: [mean, std, nobs])
        """



        if True:
            raise NotImplementedError("Biggus way of calculation is not implemented, use the dask version of the method")




        # select the interval of interest
        timesel = [i for i, d in enumerate(self.time) if start_year <= d.year <= end_year]
        data = self.data[timesel, :, :]
        times = [self.time[i] for i in timesel]


        if convert_monthly_accumulators_to_daily:
            ndays = np.array([calendar.monthrange(d.year, d.month)[1] for d in times])

            data = biggus.divide(data, ndays[:, np.newaxis, np.newaxis])

        else:
            data = self.data



        year_month_to_index_arr = defaultdict(list)
        for i, t in enumerate(times):
            year_month_to_index_arr[t.year, t.month].append(i)


        # calculate monthly means
        monthly_data = {}
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                aslice = slice(year_month_to_index_arr[y, m][0], year_month_to_index_arr[y, m][-1] + 1)
                monthly_data[y, m] = biggus.mean(data[aslice.start:aslice.stop, :, :], axis=0)


        result = {}
        for season, month_period in season_to_monthperiod.items():
            assert isinstance(month_period, MonthPeriod)

            seasonal_means = []
            ndays_per_season = []

            for p in month_period.get_season_periods(start_year=start_year, end_year=end_year):
                lmos = biggus.ArrayStack([monthly_data[start.year, start.month] for start in p.range("months")])
                ndays_per_month = np.array([calendar.monthrange(start.year, start.month)[1] for start in p.range("months")])

                seasonal_mean = biggus.sum(biggus.multiply(lmos, ndays_per_month[:, np.newaxis, np.newaxis]), axis=0)
                seasonal_mean = biggus.divide(seasonal_mean, ndays_per_month.sum())

                seasonal_means.append(seasonal_mean)
                ndays_per_season.append(ndays_per_month.sum())


            seasonal_means = biggus.ArrayStack(seasonal_means)
            ndays_per_season = np.array(ndays_per_season)


            print(seasonal_means.shape, ndays_per_season.shape)

            assert seasonal_means.shape[0] == ndays_per_season.shape[0]

            clim_mean = biggus.sum(biggus.multiply(seasonal_means, ndays_per_season[:, np.newaxis, np.newaxis]), axis=0) / ndays_per_season.sum()


            diff = biggus.subtract(seasonal_means, clim_mean.masked_array()[np.newaxis, :, :])
            sq_mean = biggus.sum(biggus.multiply(diff ** 2, ndays_per_season[:, np.newaxis, np.newaxis]), axis=0) / ndays_per_season.sum()
            clim_std = biggus.power(sq_mean, 0.5)

            clim_mean = clim_mean.masked_array()
            print("calculated mean")
            clim_std = clim_std.masked_array()
            print("calculated std")


            result[season] = [clim_mean, clim_std, ndays_per_season.shape[0]]

        return result




    def __sel_period(self, start_year, end_year, arr):
        timesel = [i for i, d in enumerate(self.time) if start_year <= d.year <= end_year]
        data = arr[timesel]
        times = [self.time[i] for i in timesel]
        return data, times





    def __get_seasmean_cache_file(self, season_to_month_period, start_year=-np.Inf, end_year=np.Inf):
        seas_tok = "_".join(season_to_month_period)
        year_tok = "{}-{}".format(start_year, end_year)

        return "DAYMET_seas__{}__{}.bin".format(seas_tok, year_tok)


    def __get_seasonal_means_with_ttest_stats_dask_lazy(self, data, season_to_monthperiod=None, start_year=-np.Inf, end_year=np.Inf,
                                                        convert_monthly_accumulators_to_daily=False):

        # mask the resulting fields
        epsilon = 1.0e-5
        mask = np.less_equal(np.abs(data[0, :, :] - self.missing_value), epsilon)

        print("data.shape = ", data.shape)

        data_sel, times_sel = data, self.time

        # select the interval of interest

        if convert_monthly_accumulators_to_daily:
            ndays = da.from_array(np.array([calendar.monthrange(d.year, d.month)[1] for d in times_sel]), (100, ))
            ndays = da.transpose(da.broadcast_to(da.from_array(ndays, ndays.shape), data_sel.shape[1:] + ndays.shape), axes=(2, 0, 1))

            data_sel = data_sel / ndays



        year_month_to_index_arr = defaultdict(list)
        for i, t in enumerate(times_sel):
            year_month_to_index_arr[t.year, t.month].append(i)

        # calculate monthly means
        monthly_data = {}
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                aslice = slice(year_month_to_index_arr[y, m][0], year_month_to_index_arr[y, m][-1] + 1)
                print(aslice, data_sel.shape)
                monthly_data[y, m] = data_sel[aslice, :, :].mean(axis=0)


        result = OrderedDict()
        for season, month_period in season_to_monthperiod.items():
            assert isinstance(month_period, MonthPeriod)

            seasonal_means = []
            ndays_per_season = []

            for p in month_period.get_season_periods(start_year=start_year, end_year=end_year):
                lmos = da.stack([monthly_data[start.year, start.month] for start in p.range("months")])
                ndays_per_month = np.array([calendar.monthrange(start.year, start.month)[1] for start in p.range("months")])
                ndays_per_month = da.from_array(ndays_per_month, ndays_per_month.shape)


                print(p)
                print(lmos.shape, ndays_per_month.shape, ndays_per_month.sum())
                seasonal_mean = da.tensordot(lmos, ndays_per_month, axes=([0,], [0,])) / ndays_per_month.sum()

                seasonal_means.append(seasonal_mean)
                ndays_per_season.append(ndays_per_month.sum())


            seasonal_means = da.stack(seasonal_means)
            ndays_per_season = np.array(ndays_per_season)
            ndays_per_season = da.from_array(ndays_per_season, ndays_per_season.shape)




            print(seasonal_means.shape, ndays_per_season.shape)

            assert seasonal_means.shape[0] == ndays_per_season.shape[0]

            clim_mean = da.tensordot(seasonal_means, ndays_per_season, axes=([0,], [0,])) / ndays_per_season.sum()

            clim_std = ((seasonal_means - da.broadcast_to(clim_mean, seasonal_means.shape)) ** 2 * ndays_per_season[:, np.newaxis, np.newaxis]).sum(axis=0) / ndays_per_season.sum()

            clim_std = clim_std ** 0.5

            result[season] = [clim_mean, clim_std, ndays_per_season.shape[0]]

        return result, mask



    def get_seasonal_means_with_ttest_stats_dask(self, season_to_monthperiod=None, start_year=-np.Inf, end_year=np.Inf,
                                            convert_monthly_accumulators_to_daily=False):


        """

        :param season_to_monthperiod:
        :param start_year:
        :param end_year:
        :param convert_monthly_accumulators_to_daily: if true converts monthly accumulators to daily,
        :return dict(season: [mean, std, nobs])
        """

        result, mask = self.__get_seasonal_means_with_ttest_stats_dask_lazy(self.data, season_to_monthperiod=season_to_monthperiod,
                                                                      start_year=start_year, end_year=end_year,
                                                                      convert_monthly_accumulators_to_daily=convert_monthly_accumulators_to_daily)

        for season in result:
            print("Computing for {}".format(season))
            for i in range(len(result[season]) - 1): # -1 because the last one is for the
                result[season][i] = np.ma.masked_where(mask, result[season][i].compute())

        return result


    def close(self):
        del self


# function applied at each gridcell to calculate the percentiles
def percentile_calculator(block, time_sel, missing_value, rolling_mean_window_days=None, percentile=0.5,
                          start_year=-np.Inf, end_year=np.Inf):

    # return the masked array if all the values for the point are masked
    """

    :param rolling_mean_window_days:
    :param percentile:
    :param time_sel: times corresponding to the first dimension of block
    :param block: 3D field (nt, nx, ny)
    :param missing_value:
    :return:
    """
    if np.all(np.less(np.abs(block[0] - missing_value), 1e-5)):
        new_shape = (365, ) + block.shape[1:]
        return missing_value * dask.array.ones(new_shape, chunks=new_shape)


    s = pd.DataFrame(data=block.reshape((len(time_sel), -1)), index=time_sel)

    s = s.select(lambda d: (not (d.month == 2 and d.day == 29)) and (start_year <= d.year <= end_year))
    assert isinstance(s, pd.DataFrame)

    if rolling_mean_window_days is not None:
        s = s.rolling(rolling_mean_window_days, center=True).mean().bfill().ffill()


    # Each group is a dataframe with the rows(axis=0) for a day of different years
    grouped = s.groupby([s.index.month, s.index.day])
    daily_perc = grouped.quantile(q=percentile)

    return daily_perc.values.reshape((-1,) + block.shape[1:])  # <- Should be (365, nx, ny)




def test():
    manager = HighResDataManager(path="/HOME/data/Validation/Daymet/Monthly_means/NetCDF/daymet_v3_prcp_monttl_*_na.nc4",
                                 vname="prcp")

    # plt.figure()
    # seas_data = manager.get_seasonal_means_with_ttest_stats({"DJF": MonthPeriod(12, 3)}, start_year=1980, end_year=1982, convert_monthly_accumulators_to_daily=True)
    seas_data = manager.get_seasonal_means_with_ttest_stats_dask({"DJF": MonthPeriod(12, 3)}, start_year=1980, end_year=1982, convert_monthly_accumulators_to_daily=True)
    # im = plt.pcolormesh(seas_data["DJF"][0])
    # plt.colorbar(im)
    # plt.show()

    manager.close()




if __name__ == '__main__':
    test()
