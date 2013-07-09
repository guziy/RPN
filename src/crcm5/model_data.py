from datetime import datetime, timedelta
import itertools
from multiprocessing import Pool
from netCDF4 import Dataset, date2num
import pickle
import shelve
import time

from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from netcdftime import num2date
from numpy.lib.function_base import meshgrid
from scipy.spatial.ckdtree import cKDTree

import application_properties
from model_point import ModelPoint
from data import cehq_station
from data.cehq_station import Station
from data.cell_manager import CellManager
from data.timeseries import DateValuePair, TimeSeries
from domains.rotated_lat_lon import RotatedLatLon
from rpn import level_kinds
from util import plot_utils, scores
from util.geo import lat_lon


__author__ = 'huziy'

import numpy as np
from rpn.rpn import RPN
import os

import matplotlib.pyplot as plt


class InputForProcessPool:
    def __init__(self):
        self.i_upstream = None
        self.j_upstream = None
        self.multipliers = None
        self.mp_ix = None
        self.mp_jy = None


def _get_var_data_to_pandas(x):
    vName, level, average_upstream, nc_sim_folder, inObject = x
    assert isinstance(inObject, InputForProcessPool)
    ds = Dataset(os.path.join(nc_sim_folder, "{0}_all.nc4".format(vName)))
    var = ds.variables[vName]
    if average_upstream:
        #data_frame[vName] = np.sum( var[:,level,:,:][:,i_upstream, j_upstream] * coefs, axis = 1 )

        imin, imax = min(inObject.i_upstream), max(inObject.i_upstream)
        jmin, jmax = min(inObject.j_upstream), max(inObject.j_upstream)

        the_data = var[:, level, imin:imax + 1, jmin:jmax + 1]

        i_adapt = np.array(inObject.i_upstream) - imin
        j_adapt = np.array(inObject.j_upstream) - jmin

        the_data = np.sum(the_data[:, i_adapt, j_adapt] * inObject.multipliers, axis=1)
    else:
        the_data = var[:, level, inObject.mp_ix, inObject.mp_jy]

    ds.close()
    return the_data

    pass


import pandas


class Crcm5ModelDataManager:
    def __init__(self, samples_folder_path="data/gemclim/quebec/Samples",
                 var_name="STFL", file_name_prefix="pm",
                 all_files_in_samples_folder=False, need_cell_manager=False):
        self.file_name_prefix = file_name_prefix
        self.samples_folder = samples_folder_path
        self.all_files_in_one_folder = all_files_in_samples_folder
        self.need_cell_manager = need_cell_manager

        #approximate estimate of the distance between grid cells
        self.characteristic_distance = None

        self._read_lat_lon_fields()
        self.run_id = "undefined"

        if not all_files_in_samples_folder:
            self._month_folder_prefix = None
            self.month_folder_name_format = "%s_%d%02d"
            self._set_month_folder_prefix()

        self._read_static_data()
        self.var_name = var_name

        self._file_paths = None

        self.name_to_date_to_field = {}
        self._flat_index_to_2d_cache = {}

        self.lon2D_rot = None
        self.lat2D_rot = None

        self.shelve_path = "cache_db"
        pass


    def get_daily_means(self, var_name=None):
        if var_name is None:
            var_name = self.var_name

        if self.all_files_in_one_folder:
            for fName in os.listdir(self.samples_folder):
                if not fName.startswith(self.file_name_prefix): continue
                fPath = os.path.join(self.samples_folder, fName)
                rObj = RPN(fPath)
                data = rObj.get_all_time_records_for_name(varname=var_name)

                rObj.close()

                #return d
        else:
            raise NotImplementedError("Output dates query is not implemented for this input")


    def _get_relevant_file_paths(self):
        paths = []
        if self.all_files_in_one_folder:
            files = os.listdir(self.samples_folder)
            for fName in files:
                if not fName.startswith(self.file_name_prefix): continue
                fPath = os.path.join(self.samples_folder, fName)
                paths.append(fPath)
        else:
            raise Exception("If you are using distributed storage, you need to implement option")
        return paths


    def _zero_negatives(self, arr):
        x = arr.copy()
        x[x < 0] = 0
        return x


    def interpolate_data_to(self, data_in, lons2d, lats2d, nneighbours=4):
        """
        Interpolates data_in to the grid defined by (lons2d, lats2d)
        assuming that the data_in field is on the initial CRU grid

        interpolate using 4 nearest neighbors and inverse of squared distance
        """

        x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons2d.flatten(), lats2d.flatten())
        dst, ind = self.kdtree.query(zip(x_out, y_out, z_out), k=nneighbours)

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


    def export_monthly_mean_fields(self, sim_name="default_sim", in_file_prefix="",
                                   start_year=1979, end_year=1988,
                                   varname="", nc_db_folder="/home/huziy/skynet3_rech1/crcm_data_ncdb",
                                   level=-1, level_kind=-1, rewrite=False):

        """
        start_year and end_year are inclusive
        save fields as a variable with the following dimensions
        F(year, month, lon, lat), file naming convention: varname.nc

        Assumes that one file contains fields for one month

        """

        nc_sim_folder = os.path.join(nc_db_folder, sim_name)
        if not os.path.isdir(nc_sim_folder):
            os.mkdir(nc_sim_folder)

        nyears = end_year - start_year + 1

        nc_path = os.path.join(nc_sim_folder, "{0}.nc4".format(varname))

        from netCDF4 import Dataset

        if os.path.isfile(nc_path) and not rewrite:
            res = raw_input("{0} already exist, do yu really want to re-export?[y/n]".format(nc_path))
        else:
            res = "y"
        if res.strip().lower() != "y":
            return
        ds = Dataset(nc_path, mode="w")

        ds.createDimension("year", nyears)
        ds.createDimension("month", 12)
        ds.createDimension("lon", self.lons2D.shape[0])
        ds.createDimension("lat", self.lons2D.shape[1])

        data = ds.createVariable(varname, "f8", dimensions=("year", "month", "lon", "lat"))
        yearVar = ds.createVariable("year", "i4", dimensions=("year",))
        lonVar = ds.createVariable("lon", "f8", dimensions=("lon", "lat"))
        latVar = ds.createVariable("lat", "f8", dimensions=("lon", "lat"))

        monthVar = ds.createVariable("month", "i4", dimensions=("month"))

        lonVar[:, :] = self.lons2D[:, :]
        latVar[:, :] = self.lats2D[:, :]
        yearVar[:] = np.arange(start_year, end_year + 1)
        monthVar[:] = np.arange(1, 13)

        if self.all_files_in_one_folder:
            for fName in os.listdir(self.samples_folder):
                if not fName.startswith(in_file_prefix): continue

                rObj = RPN(os.path.join(self.samples_folder, fName))
                rObj.suppress_log_messages()

                date_to_field = rObj.get_all_time_records_for_name_and_level(varname=varname, level=level,
                                                                             level_kind=level_kind)
                rObj.close()

                #skip the first file
                if len(date_to_field) == 1:
                    print "skipping the file corresponding to t=0."
                    continue

                dates = list(sorted(date_to_field.keys()))

                if dates[0].year < start_year or dates[0].year > end_year:
                    continue

                #assert dates[0].month == dates[-1].month, "{0} != {1}".format( str( dates[0] ),  str( dates[-1] ) )
                data[dates[0].year - start_year, dates[0].month - 1, :, :] = np.mean(date_to_field.values(), axis=0)

        ds.close()
        pass


    def get_annual_mean_fields(self, start_year=-np.Inf, end_year=np.Inf, varname=None, level=-1,
                               level_kind=level_kinds.ARBITRARY):
        """
        returns pandas.Series withe year as an index, and 2d fields of annual means as values
        {year:mean_field}
        """
        if varname is None: varname = self.var_name

        result = pandas.TimeSeries()
        paths = self._get_relevant_file_paths()

        start_date = None
        end_date = None
        if start_year != np.Inf:
            start_date = datetime(start_year, 1, 1)

        if end_year != np.Inf:
            end_date = datetime(end_year + 1, 1, 1, 0, 0)

        for aPath in paths:
            r = RPN(aPath)
            r.suppress_log_messages()
            data = r.get_all_time_records_for_name_and_level(varname=varname, level=level, level_kind=level_kind)
            ts = pandas.TimeSeries(data)
            assert isinstance(ts, pandas.Series)

            ts = ts.truncate(before=start_date, after=end_date)

            result = result.append(ts)
            r.close()

        keys_and_groups = result.groupby(by=lambda t: t.year)
        data_dict = {}
        for kg in keys_and_groups:
            key = kg[0]
            if not ( start_year <= key <= end_year): continue
            the_group = kg[1]

            print type(the_group)
            print dir(the_group)
            #the_group1 = the_group.apply(self._zero_negatives)
            data_dict[key] = the_group.mean()

        return pandas.Series(data_dict)


    def get_date_to_field_dict(self, var_name=None):
        """
        The dict, return from the method, is bounded to db
        """
        if var_name is None:
            var_name = self.var_name

        if self.all_files_in_one_folder:
            if os.path.isfile(self.shelve_path):
                os.remove(self.shelve_path)
            d = shelve.open(self.shelve_path)
            for fName in os.listdir(self.samples_folder):
                if not fName.startswith(self.file_name_prefix): continue
                fPath = os.path.join(self.samples_folder, fName)
                rObj = RPN(fPath)
                data = rObj.get_all_time_records_for_name(varname=var_name)

                keys_s = map(lambda t: t.strftime("%Y-%m-%d %H:%M"), data.keys())
                vals = map(lambda k: data[k], data.keys())

                d.update(dict(zip(keys_s, vals)))
                rObj.close()
            return d
        else:
            raise NotImplementedError("Output dates query is not implemented for this input")


    def get_lon_lat_in_rotated_coords(self, rot_lat_lon_proj):
        """
        Lame method uses loops for converting, though uses memorization in order to not repeat
        """
        assert isinstance(rot_lat_lon_proj, RotatedLatLon)

        in_shape = self.lons2D.shape

        if None not in [self.lon2D_rot, self.lat2D_rot]:
            return self.lon2D_rot, self.lat2D_rot

        self.lon2D_rot = np.zeros_like(self.lons2D)
        self.lat2D_rot = np.zeros_like(self.lons2D)

        for i in range(in_shape[0]):
            for j in range(in_shape[1]):
                self.lon2D_rot[i, j], self.lat2D_rot[i, j] = rot_lat_lon_proj.toProjectionXY(self.lons2D[i, j],
                                                                                             self.lats2D[i, j])

        return self.lon2D_rot, self.lat2D_rot

    def get_spatial_integral_over_mask_of_dyn_field(self, mask, weights_2d, path_to_folder="", var_name="",
                                                    level=-1, level_kind=level_kinds.ARBITRARY, file_prefix="dm"):
        """
        returns a timeseries object
        """
        weights_1d = weights_2d[mask == 1]
        data = {}
        for fName in os.listdir(path_to_folder):
            if not fName.startswith(file_prefix): continue
            fPath = os.path.join(path_to_folder, fName)
            rObj = RPN(fPath)
            t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level,
                                                                      level_kind=level_kind)
            for t, field in t_to_field.iteritems():
                data[t] = sum(field[mask == 1] * weights_1d)
            rObj.close()

        times = sorted(data.keys())
        vals = np.array(map(lambda t: data[t], times))
        return TimeSeries(time=times, data=vals)


    def get_dynamic_field_for_point(self, ix, jy, path_to_folder="", var_name="",
                                    level=-1, level_kind=level_kinds.ARBITRARY, file_prefix="dm"):

        """
        returns a timeseries object
        """
        data = {}
        for fName in os.listdir(path_to_folder):
            if not fName.endswith(file_prefix): continue
            fPath = os.path.join(path_to_folder, fName)
            rObj = RPN(fPath)
            t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level,
                                                                      level_kind=level_kind)
            for t, field in t_to_field.iteritems():
                data[t] = field[ix, jy]
            rObj.close()

        times = sorted(data.keys())
        vals = map(lambda t: data[t], times)
        return TimeSeries(time=times, data=vals)


    def set_projection(self, proj_obj):
        """
        set the object that can calculate areas
        """
        self.pojection = proj_obj


    def _flat_index_to_2d(self, flat_index):
        if not len(self._flat_index_to_2d_cache):
            nx, ny = self.lons2D.shape
            j2d, i2d = meshgrid(xrange(ny), xrange(nx))
            i_flat = i2d.flatten()
            j_flat = j2d.flatten()
            self._flat_index_to_2d_cache = dict(zip(xrange(len(i_flat)), zip(i_flat, j_flat)))

        return self._flat_index_to_2d_cache[flat_index]


    def get_monthly_sums_over_points(self, mask, var_name, level=-1, level_kind=level_kinds.ARBITRARY,
                                     areas2d=None, start_date=None, end_date=None):
        """
        return Timeseries object with values = sum_i( sum(data[ti][mask == 1] * Area[mask == 1])) and time step
        of the initial data, which could be useful when taking temporal integral
        Note: the result is not multiplied by timestep
        """
        areas1d = areas2d[mask == 1]

        #read in the fields
        data = {}
        if self.all_files_in_one_folder:
            for fPath in map(lambda x: os.path.join(self.samples_folder, x), os.listdir(self.samples_folder)):
                if not os.path.basename(fPath).startswith(self.file_name_prefix): continue
                rObj = RPN(fPath)
                t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level,
                                                                          level_kind=level_kind)
                for the_key in t_to_field.keys():
                    the_field = t_to_field[the_key]
                    data[the_key] = np.sum(the_field[mask == 1] * areas1d)
                rObj.close()
                pass

        dates = sorted(data.keys())
        dates = list(itertools.ifilter(lambda t: start_date <= t <= end_date, dates))
        dt = dates[1] - dates[0]
        return TimeSeries(data=map(lambda t: data[t], dates), time=dates).get_ts_of_monthly_integrals_in_time(), dt


    def get_monthly_means_over_points(self, mask, var_name, level=-1, level_kind=level_kinds.ARBITRARY,
                                      areas2d=None, start_date=None, end_date=None):
        """
        return Timeseries object with values = sum(data[ti][mask == 1] * Area[mask == 1])
        """
        areas1d = areas2d[mask == 1]

        #read in the fields
        data = {}
        if self.all_files_in_one_folder:
            fNames = os.listdir(self.samples_folder)
            for fName in fNames:
                if not fName.startswith(self.file_name_prefix): continue
                fPath = os.path.join(self.samples_folder, fName)
                rObj = RPN(fPath)
                t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level,
                                                                          level_kind=level_kind)
                for the_key in t_to_field.keys():
                    the_field = t_to_field[the_key]
                    data[the_key] = np.sum(the_field[mask == 1] * areas1d)
                rObj.close()
                pass

        dates = sorted(data.keys())
        dates = list(itertools.ifilter(lambda t: start_date <= t <= end_date, dates))
        return TimeSeries(data=map(lambda t: data[t], dates), time=dates).get_ts_of_monthly_means()


    @staticmethod
    def hdf_get_daily_climatological_fields(hdf_db_path="", var_name="", level=None):
        import tables as tb

        hdf = tb.openFile(hdf_db_path)

        varTable = hdf.getNode("/", var_name)
        assert isinstance(varTable, tb.Table)

        stamp_year = 2001  # Just select the stamp year arbitrarily  (preferably not a leap year)
        the_date = datetime(stamp_year, 1, 1)
        daily_dates = []
        day = timedelta(days=1)

        daily_fields = []
        t0 = time.clock()
        while the_date.year == stamp_year:
            if level is not None:
                expr = "(level == {0}) & (month == {1}) & (day == {2})".format(level, the_date.month, the_date.day)
                result = np.mean([row["field"] for row in varTable.where(expr)], axis=0)
            else:
                expr = "(month == {0}) & (day == {1})".format(the_date.month, the_date.day)
                result = np.mean([row["field"] for row in varTable.where(expr)], axis=0)

            daily_fields.append(result)
            daily_dates.append(the_date)
            the_date = the_date + day
            print the_date, "{0} seconds spent".format(time.clock() - t0)
        hdf.close()

        return daily_dates, daily_fields


    @staticmethod
    def hdf_get_climatology_for_season(months=None, hdf_db_path="", var_name="", level=None):
        """

        :param months: months of interest
        :param hdf_db_path: path to the hdf file
        :param var_name: name of the variable
        :param level: level of interest
        :return:
        """
        import tables as tb

        hdf = tb.openFile(hdf_db_path)

        varTable = hdf.getNode("/", var_name)
        assert isinstance(varTable, tb.Table)
        if months is None:
            months = range(1, 13)

        if level is not None:
            lev_expr = "level == {0}".format(level)
            result = np.mean([row["field"] for row in varTable.where(lev_expr) if row["month"] in months], axis=0)
        else:
            result = np.mean([row["field"] for row in varTable if row["month"] in months], axis=0)

        hdf.close()

        return result


    def export_to_hdf(self, var_list=None, file_path=""):
        """
        If var_list is None, then convert all the variables to hdf


        file -> varname -> leveltype -> Table(year, month, day, hour, minute, second, level, field)
        """
        import tables as tb


        rpn_path_list = []

        #get the list of files in rpn folder
        if self.all_files_in_one_folder:
            rpn_path_list.extend(
                [os.path.join(self.samples_folder, fName) for fName in os.listdir(self.samples_folder)])

        h5file = tb.openFile(file_path, mode="w", title="created from the data in {0}".format(self.samples_folder))


        #table row description
        class FieldDataTable(tb.IsDescription):
            year = tb.IntCol()
            month = tb.IntCol()
            day = tb.IntCol()
            hour = tb.IntCol()
            minute = tb.IntCol()
            second = tb.IntCol()

            level = tb.FloatCol()
            field = tb.FloatCol(shape=self.lons2D.shape)


        class RotatedLatlonTable(tb.IsDescription):
            name = tb.StringCol(256)
            lon1 = tb.FloatCol()
            lat1 = tb.FloatCol()
            lon2 = tb.FloatCol()
            lat2 = tb.FloatCol()


        for varName in var_list:
            varTable = h5file.createTable("/", varName, FieldDataTable)

            for fPath in rpn_path_list:
                rObj = RPN(fPath)
                rObj.suppress_log_messages()
                try:
                    data = rObj.get_4d_field(name=varName)
                except Exception:
                    #the variable not found or some other problem occurred
                    rObj.close()
                    continue

                #add the data to hdf table
                row = varTable.row
                for t, vals in data.iteritems():
                    # row = [ (t.year, t.month, t.day, t.hour, t.minute, t.second, level, field) for level, field in vals.iteritems() ]
                    for level, field in vals.iteritems():
                        row["year"] = t.year
                        row["month"] = t.month
                        row["day"] = t.day
                        row["hour"] = t.hour
                        row["minute"] = t.minute
                        row["second"] = t.second
                        row["level"] = level
                        row["field"] = field

                        row.append()
                        #close the file
                rObj.close()

        #TODO: insert also lon and lat data
        #TODO      and projection properties like /projection -> Table( name => "rotpole", "lon1" =>..., "lon2" =>)
        #TODO
        #TODO

        h5file.createArray("/", "longitude", self.lons2D)
        h5file.createArray("/", "latitude", self.lats2D)

        projTable = h5file.createTable("/", "projection_params", RotatedLatlonTable)
        row = projTable.row

        h5file.close()
        pass


    def get_daily_means_over_points(self, mask, var_name, level=-1, level_kind=level_kinds.ARBITRARY,
                                    areas2d=None, start_date=None, end_date=None):
        """
        return Timeseries object with values = sum(data[ti][mask == 1] * Area[mask == 1]) / sum(Area[mask == 1])
        """
        day = timedelta(days=1)
        the_day = datetime(start_date.year, start_date.month, start_date.day)

        areas1d = areas2d[mask == 1]
        areas_sum = np.sum(areas1d)

        #read in the fields
        data = {}
        if self.all_files_in_one_folder:
            for fPath in map(lambda x: os.path.join(self.samples_folder, x), os.listdir(self.samples_folder)):
                fName = os.path.basename(fPath)
                if not fName.startswith(self.file_name_prefix): continue

                rObj = RPN(fPath)
                t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level,
                                                                          level_kind=level_kind)
                for the_key in t_to_field.keys():
                    the_field = t_to_field[the_key]
                    data[the_key] = np.sum(the_field[mask == 1] * areas1d)
                rObj.close()
                pass

        all_dates = data.keys()
        times = []
        values = []
        while the_day <= end_date:
            times.append(the_day)
            sel_times = itertools.ifilter(lambda x: the_day.year == x.year and the_day.month == x.month and
                                                    the_day.day == x.day, all_dates)
            sel_values = map(lambda key: data[key], sel_times)
            values.append(np.mean(sel_values))
            the_day += day

        return TimeSeries(data=values, time=times)


    def _get_model_indices_for_stfl_station(self, station, nneighbours=4):
        """
        Selects a model point corresponding to the streamflow measuring station
        """
        lon, lat = station.longitude, station.latitude
        x0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        [distances, indices] = self.kdtree.query(x0, k=nneighbours)

        acc_area_1d = self.accumulation_area_km2.flatten()

        da_diff = np.abs(acc_area_1d[indices] - station.drainage_km2) / station.drainage_km2

        #don't look at the stations for which the accumulation area is too different
        if np.min(da_diff) > 0.15:
            return -1, -1

        max_dist = np.max(distances)
        obj_func = da_diff + distances / max_dist

        min_obj_func = np.min(obj_func[(acc_area_1d[indices] > 0) & (da_diff < 0.15)])
        index_of_sel_index = np.where(min_obj_func == obj_func)
        select_index = indices[index_of_sel_index[0][0]]

        print "selected index {0}".format(select_index)

        #assert np.all(acc_area_1d >= 0)
        acc_area_1d[:select_index] = -1
        acc_area_1d[select_index + 1:] = -1

        restored = np.reshape(acc_area_1d, self.accumulation_area_km2.shape)
        [i, j] = np.where(restored >= 0)

        if len(i) > 1:
            print "Warning {0} candidate cells for the station {1}".format(len(i), station.id)
        return i[0], j[0]


    def _set_metadata(self, i, j):
        """
        Set properties of the model grid cell (i,j)
        """
        return {"acc_area_km2": self.accumulation_area_km2[i, j],
                "grd_lon": self.lons2D[i, j],
                "grd_lat": self.lats2D[i, j],
                "ix": i, "jy": j,
                "bankfull_store_m3": self.bankfull_storage_m3[i, j]}


    def get_streamflow_timeseries_for_station(self, station,
                                              start_date=None, end_date=None,
                                              var_name=None, nneighbours=4):
        """
        get model data for the gridcell corresponding to the station
        :type station: data.cehq_station.Station
        :rtype: data.timeseries.Timeseries
        """

        i, j = self._get_model_indices_for_stfl_station(station, nneighbours=nneighbours)
        if i < 0: return None
        print "retrieving ts for point ({0}, {1})".format(i, j)
        ts = self.get_timeseries_for_point(i, j, start_date=start_date,
                                           end_date=end_date, var_name=var_name)

        ts.metadata = self._set_metadata(i, j)
        ts.metadata["distance_to_obs_km"] = 1.0e-3 * lat_lon.get_distance_in_meters(self.lons2D[i, j],
                                                                                    self.lats2D[i, j],
                                                                                    station.longitude, station.latitude
        )
        return ts


    def _get_any_file_path(self):
        """
        TODO: modify
        """

        if self.all_files_in_one_folder:
            return os.path.join(self.samples_folder, os.listdir(self.samples_folder)[0])

        for month_folder in os.listdir(self.samples_folder):
            month_folder_path = os.path.join(self.samples_folder, month_folder)

            if not os.path.isdir(month_folder_path):
                continue

            for the_file in os.listdir(month_folder_path):
                if the_file.startswith("."):
                    continue

                if not the_file.startswith(self.file_name_prefix):
                    continue
                return os.path.join(month_folder_path, the_file)
        pass

    def _read_lat_lon_fields(self):
        path = self._get_any_file_path()

        rpnObj = RPN(path)
        [self.lons2D, self.lats2D] = rpnObj.get_longitudes_and_latitudes()
        rpnObj.close()

        self.lons2D[self.lons2D > 180] -= 360

        #create kdtree for easier and faster lookup of the corresponding points
        #model <=> obs, for comparison
        [x, y, z] = lat_lon.lon_lat_to_cartesian(self.lons2D.flatten(), self.lats2D.flatten())
        self.kdtree = cKDTree(zip(x, y, z))


        #calculate characteristic distance
        v1 = lat_lon.lon_lat_to_cartesian(self.lons2D[0, 0], self.lats2D[0, 0])
        v2 = lat_lon.lon_lat_to_cartesian(self.lons2D[1, 1], self.lats2D[1, 1])
        dv = np.array(v2) - np.array(v1)
        self.characteristic_distance = np.sqrt(np.dot(dv, dv))
        print "Grid's approximate distance between the neighbour cells is: {0:.2f}".format(self.characteristic_distance)

        pass

    @classmethod
    def get_omerc_basemap_using_lons_lats(cls, lons2d=None, lats2d=None,
                                          lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0.0, resolution="l"
    ):

        basemap = Basemap(projection="omerc", no_rot=True, lon_1=lon_1, lat_1=lat_1,
                          lon_2=lon_2, lat_2=lat_2,
                          #width=2600000, height=2600000,
                          llcrnrlon=lons2d[0, 0], llcrnrlat=lats2d[0, 0],
                          urcrnrlon=lons2d[-1, -1], urcrnrlat=lats2d[-1, -1],
                          resolution=resolution#, lon_0 = lon_0, lat_0=lat_0
        )

        return basemap


    def get_rotpole_basemap(self, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0.0, resolution="l"):
        rll = RotatedLatLon(lon1=lon_1, lat1=lat_1, lon2=lon_2, lat2=lat_2)
        rplon, rplat = rll.get_north_pole_coords()
        lon_0, lat_0 = rll.get_true_pole_coords_in_rotated_system()

        print rplon, rplat
        print lon_0, lat_0

        basemap = Basemap(projection="rotpole", o_lon_p=rplon, o_lat_p=rplat,
                          lon_0=lon_0 - 180,
                          llcrnrlon=self.lons2D[0, 0], llcrnrlat=self.lats2D[0, 0],
                          urcrnrlon=self.lons2D[-1, -1], urcrnrlat=self.lats2D[-1, -1],
                          resolution=resolution
        )
        return basemap

    @classmethod
    def get_rotpole_basemap_using_lons_lats(cls, lons2d=None, lats2d=None, lon_1=-68, lat_1=52, lon_2=16.65,
                                            lat_2=0.0, resolution="l"):

        rll = RotatedLatLon(lon1=lon_1, lat1=lat_1, lon2=lon_2, lat2=lat_2)
        rplon, rplat = rll.get_north_pole_coords()
        lon_0, lat_0 = rll.get_true_pole_coords_in_rotated_system()

        print rplon, rplat
        print lon_0, lat_0

        basemap = Basemap(projection="rotpole", o_lon_p=rplon, o_lat_p=rplat,
                          lon_0=lon_0 - 180,
                          llcrnrlon=lons2d[0, 0], llcrnrlat=lats2d[0, 0],
                          urcrnrlon=lons2d[-1, -1], urcrnrlat=lats2d[-1, -1],
                          resolution=resolution
        )
        return basemap
        pass


    def get_omerc_basemap(self, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0, resolution="l"):
        nx, ny = self.lons2D.shape

        lon_0 = self.lons2D[nx // 2, ny // 2]
        lat_0 = self.lats2D[nx // 2, ny // 2]
        basemap = Basemap(projection="omerc", no_rot=True, lon_1=lon_1, lat_1=lat_1,
                          lon_2=lon_2, lat_2=lat_2,
                          llcrnrlon=self.lons2D[0, 0], llcrnrlat=self.lats2D[0, 0],
                          urcrnrlon=self.lons2D[-1, -1], urcrnrlat=self.lats2D[-1, -1],
                          resolution=resolution, lon_0=lon_0, lat_0=lat_0
        )
        return basemap


    def _set_month_folder_prefix(self):
        """
        determines the prefix of the folder name, which contains data
        for a given month, before the date:
        i.e. folder_name = prefix + "_" + "YYYYMM"
        """
        for fName in os.listdir(self.samples_folder):
            if fName.startswith("."):
                continue

            if "_" not in fName:
                continue

            fields = fName.split("_")
            self._month_folder_prefix = "_".join(fields[:-1])
            return self._month_folder_prefix


    def _get_start_and_end_year(self):
        """
        in case if one wants to use all the available data for calculation
        """

        all_years = []
        for file_name in os.listdir(self.samples_folder):
            all_years.append(int(file_name.split("_")[-1][:-2]))
        return min(all_years), max(all_years)


    def get_streamflow_dataframe_for_stations(self, station_list, start_date=None, end_date=None,
                                              var_name=None, nneighbours=4,
                                              distance_upper_bound_m=np.Inf):

        """
        returns pandas.DataFrame
        """
        ix_list = []
        jy_list = []
        station_to_cell_props = {}
        for s in station_list:
            assert isinstance(s, Station)
            i, j = self._get_model_indices_for_stfl_station(s, nneighbours=nneighbours)
            ix_list.append(i)
            jy_list.append(j)

            metadata = self._set_metadata(i, j)
            metadata["distance_to_obs_km"] = 1.0e-3 * lat_lon.get_distance_in_meters(self.lons2D[i, j],
                                                                                     self.lats2D[i, j],
                                                                                     s.longitude, s.latitude
            )
            station_to_cell_props[s.id] = metadata

        id_list = map(lambda s: s.id, station_list)
        df = self.get_timeseries_for_points(ix_list, jy_list, id_list, start_date=start_date, end_date=end_date,
                                            var_name=var_name)
        assert isinstance(df, pandas.DataFrame)
        print df
        return df, station_to_cell_props


    def get_timeseries_for_points(self, ix_list, jy_list, id_list,
                                  start_date=None, end_date=None,
                                  var_name=None):
        """
        returns pandas.DataFrame
        """
        if var_name is None:
            var_name = self.var_name

        if self.all_files_in_one_folder:
            fNames = os.listdir(self.samples_folder)

            fNames = itertools.ifilter(lambda name: name.startswith(self.file_name_prefix), fNames)
            fNames = list(fNames)
            paths = map(lambda x: os.path.join(self.samples_folder, x), fNames)

            times = []
            datas = []
            for the_path, fName in zip(paths, fNames):
                if os.path.isfile(the_path):
                    print "Opening the file {0} ...".format(the_path)
                    rpnObj = RPN(the_path)
                    rpnObj.suppress_log_messages()
                    hour_to_field = rpnObj.get_all_time_records_for_name(varname=var_name)

                    for t, field in hour_to_field.iteritems():
                        if start_date is not None and t < start_date:
                            continue
                        if end_date is not None and t > end_date:
                            continue
                        tmp = []
                        for i, j in zip(ix_list, jy_list):
                            if i >= 0:
                                tmp.append(field[i, j])
                            else:
                                tmp.append(np.nan)
                        times.append(t)
                        datas.append(tmp)

                    rpnObj.close()
            df = pandas.DataFrame(data=np.array(datas), index=times, columns=id_list)
            return df.sort()

            pass
        else:
            raise NotImplementedError("You need to put all files in the same folder in order to use this function")

        pass


    def get_timeseries_for_point(self, ix, iy,
                                 start_date=None, end_date=None,
                                 var_name=None, weight=1):
        """
        returns timeseries object for data: data[:, ix, iy]
        Note: uses caching in order to decrease IO operations
        :rtype: TimeSeries
        """

        if var_name is None:
            var_name = self.var_name

        dv = []

        if not self.name_to_date_to_field.has_key(var_name):
            self.name_to_date_to_field[var_name] = {}
            paths = map(lambda x: os.path.join(self.samples_folder, x), os.listdir(self.samples_folder))
            if not self.all_files_in_one_folder:
                second_level = []
                for the_path in paths:
                    if os.path.isdir(the_path):
                        second_level.extend(os.listdir(the_path))
                paths = second_level

            for the_path in paths:
                fName = os.path.basename(the_path)
                if fName.startswith(self.file_name_prefix) and os.path.isfile(the_path):
                    print "Opening the file {0} ...".format(the_path)
                    rpnObj = RPN(the_path)
                    rpnObj.suppress_log_messages()
                    hour_to_field = rpnObj.get_all_time_records_for_name(varname=var_name)
                    rpnObj.close()
                    #print( hour_to_field.items()[0][0] , "for file {0}".format(the_path))
                    self.name_to_date_to_field[var_name].update(hour_to_field)

        for time, field in self.name_to_date_to_field[var_name].iteritems():
            if start_date is not None:
                if time < start_date: continue
            if end_date is not None:
                if time > end_date: continue

            value = field[ix, iy] * weight
            dv.append(DateValuePair(date=time, value=value))

        dv.sort(key=lambda x: x.date)
        print "datei = {0}, datef = {1}".format(dv[0].date, dv[-1].date)
        return TimeSeries(data=map(lambda x: x.value, dv),
                          time=map(lambda x: x.date, dv))

        pass


    def get_monthly_climatology_of_3d_field(self, var_name="I1", file_name_prefix="pm",
                                            start_year=None, end_year=None):

        """
        assumes that each file contains month of data
        """
        filePaths = [os.path.join(self.samples_folder, name)
                     for name in os.listdir(self.samples_folder)
                     if name.startswith(file_name_prefix)]

        month_to_means = {}

        counts = np.zeros((12,))

        levels = None
        the_mean = None
        for thePath in filePaths:
            r = RPN(thePath)
            r.suppress_log_messages()
            data = r.get_4d_field(name=var_name)
            r.close()


            ##
            dates_sorted = list(sorted(data.keys()))
            the_month = dates_sorted[0].month

            month_index = the_month - 1

            the_year = dates_sorted[0].year

            if start_year is not None:
                if the_year < start_year:
                    continue

            if end_year is not None:
                if the_year > end_year:
                    continue

            if levels is None:
                levels = list(sorted(data.items()[0][1].keys()))
                nz = len(levels)
                nx, ny = data.items()[0][1].items()[0][1].shape

                the_mean = np.zeros((nx, ny, nz))
                print "levels = {0}".format(",".join([str(lev) for lev in levels]))

            #calculate mean 3d field for the month
            for k, lev in enumerate(levels):
                the_mean[:, :, k] = np.mean([fields[lev] for fields in data.values()], axis=0)


            #Calculate mean
            if counts[month_index] < 0.5:
                month_to_means[month_index] = the_mean
            else:
                cur_mean = month_to_means[month_index]
                month_to_means[month_index] = ( cur_mean * counts[month_index] + the_mean) / (counts[month_index] + 1.0)

            counts[month_index] += 1

        return month_to_means


    def get_mean_field(self, start_year, end_year, months=None, file_name_prefix="pm",
                       var_name="STFL", level=-1, level_kind=level_kinds.ARBITRARY):
        if self.all_files_in_one_folder:
            fNames = itertools.ifilter(lambda theName: theName.startswith(file_name_prefix),
                                       os.listdir(self.samples_folder))
            fNames = list(fNames)
            fPaths = map(lambda x: os.path.join(self.samples_folder, x), fNames)

            fields_list = []
            for fPath in fPaths:
                rObj = RPN(fPath)
                rObj.suppress_log_messages()
                data = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level,
                                                                    level_kind=level_kind)
                for t, field in data.iteritems():
                    if start_year <= t.year <= end_year:
                        if t.month in months:
                            fields_list.append(field)
                rObj.close()
            return np.mean(fields_list, axis=0)
        else:
            raise NotImplementedError(
                "Need to implement the case of this data organization, or put all the data files to the same folder.")

        pass

    def get_monthly_mean_fields(self, start_date=None,
                                end_date=None,
                                months=xrange(1, 13),
                                var_name="STFL",
                                file_name_prefix="pm",
                                level=-1,
                                level_kind=level_kinds.ARBITRARY
    ):
        """
        get list of monthly means over data
        corresponding to t: start_date <= t <= end_date
        returns a map {month_number : mean_value_2d_field}
        file_name_prefix = can be "pm" or "dp"

        :type start_date: datetime.datetime or None
        :type end_date: datetime.datetime or None
        """

        [start_year, end_year] = self._get_start_and_end_year()

        if start_date is not None:
            start_year = start_date.year

        if end_date is not None:
            end_year = end_date.year

        field_list = []

        result = {}
        for m in months:
            result[m] = []

        if self.all_files_in_one_folder:
            for fName in os.listdir(self.samples_folder):
                fPath = os.path.join(self.samples_folder, fName)
                pass

        for the_year in xrange(start_year, end_year + 1):
            for the_month in months:
                folder_name = self.month_folder_name_format % (self._month_folder_prefix, the_year, the_month)
                folder_path = os.path.join(self.samples_folder, folder_name)
                for file_name in os.listdir(folder_path):
                    if file_name.startswith("."): continue
                    if not file_name.startswith(file_name_prefix): continue
                    file_path = os.path.join(folder_path, file_name)
                    field_list.append(self._get_2d_field_from_file(path_to_file=file_path,
                                                                   field_name=var_name, level=level,
                                                                   level_kind=level_kind))

                #store mean field for the given month and year
                result[the_month].append(np.mean(field_list, axis=0))

        for m in months:
            result[m] = np.mean(result[m], axis=0)

        return result
        pass

    def _get_2d_field_from_file(self, path_to_file="",
                                field_name="STFL", level=-1, level_kind=level_kinds.ARBITRARY):
        """
        Read 2D data field from a file
        """
        rpnObj = RPN(path=path_to_file)
        data = rpnObj.get_first_record_for_name_and_level(varname=field_name, level=level, level_kind=level_kind)
        rpnObj.close()
        return data


    def get_monthly_climatology(self, varname="STFL", months=range(1, 13), level=-1,
                                level_kind=level_kinds.ARBITRARY, start_date=None, end_date=None):
        time_series = {}
        if self.all_files_in_one_folder:
            for fName in os.listdir(self.samples_folder):
                fPath = os.path.join(self.samples_folder, fName)
                rpnObj = RPN(fPath)
                rpnObj.suppress_log_messages()
                date_to_field = rpnObj.get_all_time_records_for_name_and_level(varname=varname, level=level,
                                                                               level_kind=level_kind)
                time_series.update(date_to_field)

        result = []

        if start_date is None:
            start_date = min(time_series.keys())

        if end_date is None:
            end_date = max(time_series.keys())

        for month in months:
            sel_dates = itertools.ifilter(lambda x: x.month == month and start_date <= x <= end_date,
                                          time_series.keys())
            sel_values = map(lambda x: time_series[x], sel_dates)
            the_mean = np.array(sel_values).mean(axis=0)
            result.append(the_mean)

        return result


    def get_mean_2d_field_and_all_data(self, var_name="STFL", level=-1,
                                       level_kind=level_kinds.ARBITRARY):
        all_fields = []
        date_to_field = {}
        if self.all_files_in_one_folder:
            for fName in os.listdir(self.samples_folder):
                if not fName.startswith(self.file_name_prefix): continue
                fPath = os.path.join(self.samples_folder, fName)
                rpnObj = RPN(fPath)
                cur_date_to_field = rpnObj.get_all_time_records_for_name_and_level(
                    varname=var_name, level=level, level_kind=level_kind)
                all_fields.extend(cur_date_to_field.values())
                date_to_field.update(cur_date_to_field)

        return np.mean(all_fields, axis=0), date_to_field


    def get_mean_in_time(self, var_name="STFL"):
        all_fields = []
        if self.all_files_in_one_folder:
            for fName in os.listdir(self.samples_folder):
                fPath = os.path.join(self.samples_folder, fName)
                rpnObj = RPN(fPath)
                all_fields.extend(rpnObj.get_all_time_records_for_name(varname=var_name).values())
        return np.mean(all_fields, axis=0)


    def _read_static_field(self, rpnObj, varname):
        msg = "{0} variable was not found in the dataset"
        res = None
        try:
            res = rpnObj.get_first_record_for_name(varname)
        except Exception, e:
            print msg.format(varname)

        return res


    def _read_static_data(self, derive_from_data=True):
        """
         get drainage area fields

        """

        if derive_from_data and self.all_files_in_one_folder:
            files = os.listdir(self.samples_folder)
            files = itertools.ifilter(lambda x: not x.startswith(".") and x.startswith("pm"), files)
            file = sorted(files, key=lambda x: x[2:])[0] #take the first file disregarding prefixes
            file_path = os.path.join(self.samples_folder, file)
            rpnObj = RPN(file_path)

            varname = "FAA"
            self.accumulation_area_km2 = self._read_static_field(rpnObj, varname)

            varname = "FLDR"
            self.flow_directions = self._read_static_field(rpnObj, varname)

            varname = "ML"
            self.lake_fraction = self._read_static_field(rpnObj, varname)

            varname = "STBM"
            self.bankfull_storage_m3 = self._read_static_field(rpnObj, varname)

            varname = "CBF"
            self.cbf = self._read_static_field(rpnObj, varname)

            varname = "GWRT"
            self.gw_res_time = self._read_static_field(rpnObj, varname)

            varname = "LKAR"
            self.lake_area = self._read_static_field(rpnObj, varname)

            #try to read cell areas from the input file
            varname = "DX"
            self.cell_area = self._read_static_field(rpnObj, varname) #in m**2

            varname = "MABF"
            self.manning_bf = self._read_static_field(rpnObj, varname) #in m**2

            varname = "MG"
            self.mg = self._read_static_field(rpnObj, varname)
            self.land_sea_mask = (self.mg > 0.6).astype(int) #0/1

            varname = "SLOP"
            self.slope = self._read_static_field(rpnObj, varname)

            varname = "LKOU"
            self.lkou = self._read_static_field(rpnObj, varname)

            if self.need_cell_manager:
                nx, ny = self.flow_directions.shape
                self.cell_manager = CellManager(nx, ny, self.flow_directions)



            #self.cbf = rpnObj.get_first_record_for_name("CBF")
            rpnObj.close()
            #self.slope = rpnObj.get_first_record_for_name("SLOP")
            return

        if derive_from_data and not self.all_files_in_one_folder:
            month_folders = os.listdir(self.samples_folder)
            month_folders = sorted(month_folders)
            month_folders = map(lambda x: os.path.join(self.samples_folder, x), month_folders)
            month_folders = itertools.ifilter(lambda x: os.path.isdir(x), month_folders)

            month_folders = list(month_folders)

            files = os.listdir(month_folders[0])
            files = itertools.ifilter(lambda x: x.startswith(self.file_name_prefix)
            and not x.startswith("."), files
            )
            file = sorted(files)[0]
            file_path = os.path.join(month_folders[0], file)
            rpnObj = RPN(file_path)
            self.accumulation_area_km2 = rpnObj.get_first_record_for_name("FAA")
            self.flow_directions = rpnObj.get_first_record_for_name("FLDR")
            self.slope = rpnObj.get_first_record_for_name("SLOP")

            rpnObj.close()
            #self.slope = rpnObj.get_first_record_for_name("SLOP")
            return


        ##
        file_path = os.path.join(self.samples_folder, "..")
        file_path = os.path.join(file_path, "infocell.rpn")
        rpnObj = RPN(file_path)
        #TODO: Custom margins, fix it
        self.accumulation_area_km2 = rpnObj.get_first_record_for_name("FACC")[10:-10, 10:-10]
        self.slope = rpnObj.get_first_record_for_name("SLOP")[10:-10, 10:-10]
        rpnObj.close()

        pass


    def get_timeseries_for_station(self, var_name="", station=None,
                                   nneighbours=1, start_date=None, end_date=None):
        """
        :rtype: list of  TimeSeries  or None
        """
        lon, lat = station.longitude, station.latitude
        x0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        [distances, indices] = self.kdtree.query(x0, k=nneighbours)

        print indices, distances

        all_ts = []

        if nneighbours == 1:
            indices = [indices]

        lf_norm = 0.0
        for the_index in indices:
            ix, jy = self._flat_index_to_2d(the_index)
            if self.lake_fraction[ix, jy] <= 0.01: continue
            lf_norm += self.lake_fraction[ix, jy]

        if lf_norm <= 0.01: return None
        for the_index in indices:
            ix, jy = self._flat_index_to_2d(the_index)
            if self.lake_fraction[ix, jy] <= 0.01: continue
            ts = self.get_timeseries_for_point(ix, jy, var_name=var_name,
                                               weight=self.lake_fraction[ix, jy] / lf_norm, start_date=start_date,
                                               end_date=end_date
            )
            all_ts.append(ts)

        return all_ts

    def get_mask_for_cells_upstream(self, i_model0, j_model0):
        #cache the list of upstream cells for performance
        if hasattr(self, "upstream_cells_cache"):
            if self.upstream_cells_cache.has_key((i_model0, j_model0)):
                return self.upstream_cells_cache[(i_model0, j_model0)]
        else:
            self.upstream_cells_cache = {}
            ######
        if self.cell_manager is not None:
            aCell = self.cell_manager.cells[i_model0][j_model0]
            self.upstream_cells_cache[(i_model0, j_model0)] = self.cell_manager.get_mask_of_cells_connected_with(aCell)
        else:
            print "CellManager should be supplied for this operation."
            raise Exception("Cellmanager should be created via option need_cell_manager = True in the constructor.")

        return self.upstream_cells_cache[(i_model0, j_model0)]


    def get_2d_field_for_date(self, date, dateo, varname, level=-1, level_kind=level_kinds.ARBITRARY):
        if self.all_files_in_one_folder:
            fNames = os.listdir(self.samples_folder)
            paths = map(lambda x, y: os.path.join(x, y), [self.samples_folder] * len(fNames), fNames)
            for thePath in paths:
                rObj = RPN(thePath)
                rObj.suppress_log_messages()

                #data = rObj.get_all_time_records_for_name_and_level(varname=varname, level = level, level_kind= level_kind)
                data = rObj.get_record_for_date_and_level(var_name=varname, level=level, date=date, date_o=dateo,
                                                          level_kind=level_kind)

                rObj.close()

                #if the field for date is found
                if data is not None:
                    return data
        else:
            raise NotImplementedError("You need to implement this method for the given positions of files")


    def _create_nc_structure(self, dataset, levels, var_name):
        dataset.createDimension("time", None)
        dataset.createDimension("lon", self.lons2D.shape[0])
        dataset.createDimension("lat", self.lons2D.shape[1])
        dataset.createDimension("level", len(levels))

        dataVar = dataset.createVariable(var_name, "f8", dimensions=("time", "level", "lon", "lat"))
        lonVar = dataset.createVariable("lon", "f8", dimensions=("lon", "lat"))
        latVar = dataset.createVariable("lat", "f8", dimensions=("lon", "lat"))

        levelVar = dataset.createVariable("level", "f8", dimensions=("level", ))
        timeVar = dataset.createVariable("time", "f8", dimensions=("time", ))

        levelVar[:] = levels
        lonVar[:, :] = self.lons2D[:, :]
        latVar[:, :] = self.lats2D[:, :]
        return dataVar


    def export_field_to_netcdf(self, start_year, end_year, nc_sim_folder="", var_name=""):
        """
        Exports data to netcdf without aggregation
        nc_sim_folder - is a folder with netcdf files for a given simulation
        the file name patterns are <varname>_all.nc4

        the data for the period [start_year, end_year] should exist


        Note: the dates are not continuous here and not growing monotonously
        """

        nc_file_path = "{0}_all.nc4".format(var_name)

        nc_file_path = os.path.join(nc_sim_folder, nc_file_path)
        start_date = datetime(start_year, 1, 1)

        if os.path.isfile(nc_file_path):
            res = raw_input("The file {0} already exists, do you want to reexport? [y/n]: ".format(nc_file_path))
            if res.strip().lower() != "y":
                return

        if self.all_files_in_one_folder:
            ds = Dataset(nc_file_path, "w")

            dataVar = None
            timeVar = None
            levels = None
            t = 0

            fNames = os.listdir(self.samples_folder)
            fNames = itertools.ifilter(lambda name: name.startswith(self.file_name_prefix), fNames)
            fNames = sorted(fNames, key=lambda name: int(name.split("_")[-1][:-1]))

            fNames = list(fNames)

            for fName in fNames:
                fPath = os.path.join(self.samples_folder, fName)

                rObj = RPN(fPath)
                #{time => {level => F(x, y)}}
                rObj.suppress_log_messages()
                data = rObj.get_4d_field(name=var_name)

                rObj.close()

                if dataVar is None:
                    levels = data.items()[0][1].keys()
                    levels = list(sorted(levels))

                    dataVar = self._create_nc_structure(ds, levels, var_name)
                    timeVar = ds.variables["time"]
                    timeVar.units = "hours since {0}".format(str(start_date))

                #put data and time to netcdf file
                times = sorted(data.keys())
                times = list(itertools.ifilter(lambda d: d.year <= end_year, times))


                #if there are records in the given time range
                if len(times) > 0:
                    times_num = date2num(times, units=timeVar.units)
                    timeVar[t:] = times_num[:]
                    for k, level in enumerate(levels):
                        dataVar[t:, k, :, :] = np.array(
                            [data[d][level] for d in times]
                        )

                    t += len(times) #remember how many time steps have already been written

            ds.close()
        else:
            raise Exception("Not yet implemented")

        pass


    def export_daily_mean_fields(self, start_year, end_year, var_name="", nc_sim_folder="", quiet=False):
        """
        Exports daily mean fields to netcdf
        nc_sim_folder - is a folder with netcdf files for a given simulation
        the file name patterns are <varname>_daily.nc

        """
        nc_file_name = "{0}_daily.nc".format(var_name)

        nc_file_path = os.path.join(nc_sim_folder, nc_file_name)
        start_date = datetime(start_year, 1, 1)

        if os.path.isfile(nc_file_path) and not quiet:
            res = raw_input("The file {0} already exists, do you want to reexport? [y/n]: ".format(nc_file_path))
            if res.strip().lower() != "y":
                return

        if self.all_files_in_one_folder:
            ds = Dataset(nc_file_path, "w", format="NETCDF3_CLASSIC")

            dataVar = None
            timeVar = None
            levels = None
            t = 0

            fNames = os.listdir(self.samples_folder)
            fNames = itertools.ifilter(lambda name: name.startswith(self.file_name_prefix), fNames)
            fNames = sorted(fNames, key=lambda name: int(name.split("_")[-1][:-1]))

            fNames = list(fNames)

            for fName in fNames:
                fPath = os.path.join(self.samples_folder, fName)

                rObj = RPN(fPath)
                #{time => {level => F(x, y)}}
                rObj.suppress_log_messages()
                data = rObj.get_4d_field(name=var_name)

                #TODO: calculate daily means

                rObj.close()

                if dataVar is None:
                    levels = data.items()[0][1].keys()
                    levels = list(sorted(levels))

                    dataVar = self._create_nc_structure(ds, levels, var_name)
                    timeVar = ds.variables["time"]
                    timeVar.units = "hours since {0}".format(str(start_date))

                #put data and time to netcdf file
                times = sorted(data.keys())
                times = list(itertools.ifilter(lambda d: start_year <= d.year <= end_year, times))


                #if there are records in the given time range
                if len(times) > 0:
                    times_num = date2num(times, units=timeVar.units)
                    timeVar[t:] = times_num[:]
                    for k, level in enumerate(levels):
                        dataVar[t:, k, :, :] = np.array(
                            [data[d][level] for d in times]
                        )

                    t += len(times) #remember how many time steps have already been written

            ds.close()
        else:
            raise Exception("Not yet implemented")

        pass
        pass


    def _init_model_point(self, station, ix, jy, dist_to_station, timeArr, nc_sim_folder):
        assert isinstance(station, Station)

        #variables to be read and correspondong levels
        varnames = ["STFL", "I5", "TT", "PR", "TRAF", "TDRA"]
        levels = [0, 0, 0, 0, 4, 4] #zero-based
        mean_upstream = [False, True, True, True, True, True]

        mp = ModelPoint()
        mp.accumulation_area = self.accumulation_area_km2[ix, jy]
        mp.lake_fraction = self.lake_fraction[ix, jy]
        mp.ix = ix
        mp.jy = jy
        mp.longitude = self.lons2D[mp.ix, mp.jy]
        mp.latitude = self.lats2D[mp.ix, mp.jy]
        mp.distance_to_station = dist_to_station

        #flow in mask
        mp.flow_in_mask = self.get_mask_for_cells_upstream(ix, jy)

        mp.continuous_data_years = station.get_list_of_complete_years()
        mp.mean_upstream_lake_fraction = self.lake_fraction[mp.flow_in_mask == 1].mean()

        inObject = InputForProcessPool()
        inObject.mp_ix = mp.ix
        inObject.mp_jy = mp.jy

        inObject.i_upstream, inObject.j_upstream = np.where(mp.flow_in_mask == 1)
        sel_areas = self.cell_area[inObject.i_upstream, inObject.j_upstream]
        inObject.multipliers = sel_areas / np.sum(sel_areas)

        t0 = time.time()

        #vName, level, average_upstream, nc_sim_folder, inObject = x
        nvars = len(varnames)

        if not hasattr(self, "reusable_pool"):
            self.reusable_pool = Pool(processes=nvars)
        ppool = self.reusable_pool

        frames = ppool.map(_get_var_data_to_pandas,
                           zip(varnames, levels, mean_upstream, [nc_sim_folder, ] * nvars, [inObject, ] * nvars))

        print "extracted data from netcdf in {0} s".format(time.time() - t0)

        data_frame = pandas.DataFrame(index=timeArr)
        for vName, frame in zip(varnames, frames):
            data_frame[vName] = frame

        data_frame["year"] = data_frame.index.map(lambda d: d.year)
        #select only the years that the complete timeseries exist for the station corresponding  to the model point
        data_frame = data_frame.drop(data_frame.index[~data_frame.year.isin(station.get_list_of_complete_years())])
        data_frame = data_frame.resample("D", how="mean")

        assert isinstance(data_frame, pandas.DataFrame)
        data_frame = data_frame.groupby(lambda d: (d.day, d.month )).mean()
        mp.climatology_data_frame = data_frame[
            data_frame.index.map(lambda tup: tup != (29, 2))] #select all except 29 of Feb

        return mp


    def get_dataless_model_points_for_stations(self, station_list):
        """
        returns a map {station => modelpoint} for comparison modeled streamflows with observed

        this uses exactly the same method for searching model points as one in diagnose_point (nc-version)

        """
        model_acc_area = self.accumulation_area_km2
        model_acc_area_1d = model_acc_area.flatten()

        nx, ny = model_acc_area.shape
        t0 = time.time()
        npoints = 1
        result = {}
        for s in station_list:
            #list of model points which could represent the station

            assert isinstance(s, Station)
            x, y, z = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)
            dists, inds = self.kdtree.query((x, y, z), k=npoints)

            if npoints == 1:
                dists = np.array([dists])
                inds = np.array([inds], dtype=int)

                deltaDaMin = np.min(np.abs(model_acc_area_1d[inds] - s.drainage_km2))

                #this returns a  list of numpy arrays
                imin = np.where(np.abs(model_acc_area_1d[inds] - s.drainage_km2) == deltaDaMin)[0][0]

                deltaDa2D = np.abs(self.accumulation_area_km2 - s.drainage_km2)

                ij = np.where(deltaDa2D == deltaDaMin)

                ix, jy = ij[0][0], ij[1][0]

                #check if it is not global lake cell
                if self.lake_fraction[ix, jy] >= 0.6:
                    continue

                #check if the gridcell is not too far from the station
                if dists[imin] > 2 * self.characteristic_distance:
                    continue

                #check if difference in drainage areas is not too big less than 10 %
                if deltaDaMin / s.drainage_km2 > 0.1: continue

                mp = ModelPoint()
                mp.accumulation_area = self.accumulation_area_km2[ix, jy]
                mp.lake_fraction = self.lake_fraction[ix, jy]
                mp.ix = ix
                mp.jy = jy
                mp.longitude = self.lons2D[mp.ix, mp.jy]
                mp.latitude = self.lats2D[mp.ix, mp.jy]
                mp.distance_to_station = dists[imin]

                #flow in mask
                mp.flow_in_mask = self.get_mask_for_cells_upstream(ix, jy)
                result[s] = mp
            else:
                raise Exception("npoints = {0}, is not yet implemented ...")
        return result

        pass


    def get_model_points_for_stations(self, station_list, sim_name="", nc_path="",
                                      npoints=1, nc_sim_folder=None):
        """
        returns a map {station => [modelpoint1, ...]} for comparison modeled streamflows with observed
        modelpoint.data - contains series of data in time for the modelpoint
        modelpoint.time - contains times corresponding to modelpoint.data
        """

        cache_file = "stations_to_model_points_{0}.bin".format(sim_name)
        if os.path.isfile(cache_file):
            print "I found a cache file: {0}, and will be using data from it, if you want to reimport the data, delete the cache file".format(
                cache_file)
            return pickle.load(open(cache_file))

        station_to_grid_point = {}
        model_acc_area = self.accumulation_area_km2
        model_acc_area_1d = model_acc_area.flatten()

        nx, ny = model_acc_area.shape

        jy_indices_2d, ix_indices_2d = np.meshgrid(range(ny), range(nx))
        jy_indices_flat = jy_indices_2d.flatten()
        ix_indices_flat = ix_indices_2d.flatten()

        ds = Dataset(nc_path)
        timeVar = ds.variables["time"]
        timeArr = num2date(timeVar[:], timeVar.units)
        ds.close()

        t0 = time.time()

        for s in station_list:
            #list of model points which could represent the station
            mp_list_for_station = []

            assert isinstance(s, Station)
            x, y, z = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)
            dists, inds = self.kdtree.query((x, y, z), k=8)

            if npoints == 1:

                deltaDaMin = np.min(np.abs(model_acc_area_1d[inds] - s.drainage_km2))

                #this returns a  list of numpy arrays
                imin = np.where(np.abs(model_acc_area_1d[inds] - s.drainage_km2) == deltaDaMin)[0][0]

                deltaDa2D = np.abs(self.accumulation_area_km2 - s.drainage_km2)

                ij = np.where(deltaDa2D == deltaDaMin)

                #check if it is not global lake cell
                if self.lake_fraction[ij[0][0], ij[1][0]] >= 0.6:
                    continue

                #check if the gridcell is not too far from the station
                if dists[imin] > 2 * self.characteristic_distance:
                    continue

                #check if difference in drainage areas is not too big less than 10 %
                if deltaDaMin / s.drainage_km2 > 0.1: continue

                mp = self._init_model_point(s, ij[0][0], ij[1][0], dists[imin], timeArr, nc_sim_folder)
                mp_list_for_station.append(mp)

            else:
                for d, i in zip(dists, inds):
                    mp = self._init_model_point(s, ix_indices_flat[i], jy_indices_flat[i], d, timeArr, nc_sim_folder)
                    mp_list_for_station.append(mp)

            #if no model points found for the station, do not put it in the result dictionary
            if not len(mp_list_for_station):
                continue
            station_to_grid_point[s] = mp_list_for_station

        print "read in data from netcdf in {0} seconds".format(time.time() - t0)


        #save to cache file
        pickle.dump(station_to_grid_point, open(cache_file, mode="w"))

        return station_to_grid_point


    def get_list_of_ts_from_netcdf(self, nc_path="", varname="STFL", ix_list=None, jy_list=None):
        ds = Dataset(nc_path)
        var = ds.variables[varname][:]

        var = var[:, :, np.array(ix_list), np.array(jy_list)]

        print var.shape

        tVar = ds.variables["time"]
        times = num2date(tVar[:], tVar.units)

        ds.close()
        return times, var

        pass


def do_test_seasonal_mean():
    fig = plt.figure()
    assert isinstance(fig, Figure)
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1"
    manager = Crcm5ModelDataManager(samples_folder_path=data_path, all_files_in_samples_folder=True

    )

    gs = gridspec.GridSpec(4, 3)

    basemap = Basemap(projection="omerc", no_rot=True, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0,
                      llcrnrlon=manager.lons2D[0, 0], llcrnrlat=manager.lats2D[0, 0],
                      urcrnrlon=manager.lons2D[-1, -1], urcrnrlat=manager.lats2D[-1, -1],
                      resolution="l"
    )

    [x, y] = basemap(manager.lons2D, manager.lats2D)

    colormap = cm.get_cmap("Blues") #my_colormaps.get_red_blue_colormap(10)

    month_clim = manager.get_monthly_climatology(varname="STFL")
    the_mean = np.mean(month_clim, axis=0)
    cond = the_mean > 25
    for i, field in enumerate(month_clim):
        #ax = fig.add_subplot(gs[i // 3, i % 3])
        fig = plt.figure()
        ax = fig.gca()
        d = datetime(2000, i + 1, 1)
        ax.set_title(d.strftime("%B"))

        data = np.ma.masked_all(the_mean.shape)
        data[cond] = (field[cond] - the_mean[cond]) / the_mean[cond] * 100.0
        #data = np.ma.masked_where( ~cond, data )
        print np.ma.min(data), np.ma.max(data)
        #data = np.ma.log(data)
        img = basemap.pcolormesh(x, y, data, ax=ax,
                                 vmax=100.0, vmin=-100, cmap=colormap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(img, ax=ax, cax=cax, extend="both")
        basemap.drawcoastlines(linewidth=0.5, ax=ax)
        plt.savefig("season_{0:02d}.png".format(d.month))
        pass


    #plt.pcolormesh(manager.get_monthly_mean_fields(months = [6])[6].transpose())


    #get ocean mask
    #lons2D = manager.lons2D[:,:]
    #lons2D[lons2D >= 180] -= 360.0
    #ocean_mask = maskoceans(lons2D, manager.lats2D, data)


    #data = np.ma.masked_where(data < 0.1, data)


    plt.savefig("mean_clim.png")
    pass


def do_test_mean():
    plt.figure()
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1"
    manager = Crcm5ModelDataManager(samples_folder_path=data_path, all_files_in_samples_folder=True

    )


    #plt.pcolormesh(manager.get_monthly_mean_fields(months = [6])[6].transpose())

    basemap = Basemap(projection="omerc", no_rot=True, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0,
                      llcrnrlon=manager.lons2D[0, 0], llcrnrlat=manager.lats2D[0, 0],
                      urcrnrlon=manager.lons2D[-1, -1], urcrnrlat=manager.lats2D[-1, -1],
                      resolution="l"
    )

    [x, y] = basemap(manager.lons2D, manager.lats2D)
    data = manager.get_mean_in_time(var_name="STFL")

    #get ocean mask
    #lons2D = manager.lons2D[:,:]
    #lons2D[lons2D >= 180] -= 360.0
    #ocean_mask = maskoceans(lons2D, manager.lats2D, data)

    data = np.ma.masked_where((data < 50), data)

    #data = np.ma.masked_where(data < 0.1, data)
    basemap.pcolormesh(x, y, data)
    plt.colorbar()
    basemap.drawcoastlines(linewidth=0.5)

    plt.savefig("mean.png")
    pass


def compare_lake_levels():
    #lake level controlled only with evaporation and precipitation
    data_path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-hcd-rl_spinup"

    #lake level controlled only by routing
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1"

    selected_ids = [
        "093807", "011508", "061303", "040408", "030247"
    ]

    coord_file = os.path.join(data_path, "pm1979010100_00000000p")

    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
                                    file_name_prefix="pm", all_files_in_samples_folder=True, var_name="CLDP"
    )

    start_date = datetime(1979, 1, 1)
    end_date = datetime(1988, 12, 31)

    stations = cehq_station.read_station_data(folder="data/cehq_levels",
                                              start_date=start_date, end_date=end_date, selected_ids=selected_ids
    )

    plot_utils.apply_plot_params(width_pt=None, height_cm=30.0, width_cm=16, font_size=10)

    ns = len(stations)
    ncols = 2
    nrows = ns // ncols + 1
    plot_utils.apply_plot_params(width_pt=None, font_size=10)
    gs = gridspec.GridSpec(nrows, ncols)
    fig = plt.figure()
    assert isinstance(fig, Figure)
    sub_p = 0
    h_m = None
    h_s = None
    r = 0
    c = 0
    selected_stations = []
    for i, s in enumerate(sorted(stations, key=lambda x: x.latitude, reverse=True)):


        assert isinstance(s, Station)
        mod_ts_all = manager.get_timeseries_for_station(var_name="CLDP", station=s, nneighbours=1,
                                                        start_date=start_date, end_date=end_date
        )
        if mod_ts_all is None: #means model does not see lakes in the vicinity
            continue
        print(s.id, ":", s.get_timeseries_length())

        mod_normals_all = []

        for mod_ts in mod_ts_all:
            mod_day_dates, mod_normals = mod_ts.get_daily_normals(start_date=start_date, end_date=end_date,
                                                                  stamp_year=2001)
            mod_normals_all.append(mod_normals)
        mod_normals = np.mean(mod_normals_all, axis=0)

        if np.max(mod_normals) < 0.1: continue

        sta_day_dates, sta_normals = s.get_daily_normals(start_date=start_date, end_date=end_date, stamp_year=2001)

        if None in [sta_day_dates, sta_normals]:
            continue

        r = sub_p // ncols
        c = sub_p % ncols
        sub_p += 1
        ax = fig.add_subplot(gs[r, c])
        assert isinstance(ax, Axes)
        print len(mod_normals), len(sta_day_dates)
        #normals
        h_m = ax.plot(sta_day_dates, mod_normals - np.mean(mod_normals), "b", label="model", lw=3)
        h_s = ax.plot(sta_day_dates, sta_normals - np.mean(sta_normals), "r", label="station", lw=2)

        #instantaneous values
        #h_m = ax.plot(mod_ts_all[0].time, mod_ts_all[0].data - np.mean(mod_ts_all[0].data) , "b", label = "model", lw = 3)
        #h_s = ax.plot(s.dates, s.values - np.mean(s.values), "r", label = "station", lw = 1)



        ax.set_title(s.id + "({0:.3f}, {1:.3f})".format(s.longitude, s.latitude))
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=15, interval=2))

        ax.yaxis.set_major_locator(LinearLocator(numticks=5))

        selected_stations.append(s)
        #ax.legend()

    ax = fig.add_subplot(gs[(r + 1):, :])
    #lons2d, lats2d = manager.lons2D, manager.lats2D
    b = manager.get_rotpole_basemap()
    b.drawcoastlines()
    y1 = 0.8
    dy = y1 / float(len(selected_stations))

    x1 = [0.05, 0.85]
    for i, s in enumerate(selected_stations):
        x, y = b(s.longitude, s.latitude)
        b.scatter(x, y, c="r", marker="^", zorder=5, s=100)
        xy_text = x1[i % 2], y1
        y1 -= dy
        ax.annotate(s.id, xy=(x, y), xytext=xy_text, textcoords="axes fraction",
                    bbox=dict(facecolor='white'), weight="bold",
                    arrowprops=dict(facecolor='black', arrowstyle="->")
        )

    fig.legend([h_m[0], h_s[0]], ["Model", "Obs."], "lower right")
    fig.tight_layout(h_pad=2)
    fig.savefig("lake_level_comp_mean_anomalies.png")


def _get_cache_file_name(i, j, var_name):
    pass


def get_timeseries_from_crcm4_for_station(station):
    pass


def compare_streamflow_normals():
    #lake level controlled only by routing
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1"

    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1_lakes_off_high_res"
    #coord_file = os.path.join(data_path, "pm1985050100_00000000p")


    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_198501_198612_0.1deg"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_88x88_0.5deg_with_lakes"

    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_with_lakes"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_88x88_0.5deg_with_lakes"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_without_lakes"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_lakes_flake"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_without_lakes_v3"

    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3_sturm_snc"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_lakes_and_lakerof"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3_old_snc"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_260x260_wo_lakes_and_with_lakeroff"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types_crcm_lk_fractions"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_river_ice_1yrspnp_const_manning"
    data_path = "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff"
    coord_file = os.path.join(data_path, "pm1985050100_00000000p")

    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
                                    file_name_prefix="pm", all_files_in_samples_folder=True
    )
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                    "081006", "040830"]

    start_date = datetime(1986, 1, 1)
    end_date = datetime(1990, 12, 31)

    selected_ids = None
    stations = cehq_station.read_station_data(selected_ids=selected_ids,
                                              start_date=start_date, end_date=end_date
    )

    station_to_model_ts = {}
    for s in stations:
        assert isinstance(s, Station)

        #skip stations with smaller accumulation areas
        if s.drainage_km2 <= 2 * np.radians(0.5) ** 2 * lat_lon.EARTH_RADIUS_METERS ** 2 * 1.0e-6:
            continue

        if not s.passes_rough_continuity_test(start_date, end_date):
            continue

        model_ts = manager.get_streamflow_timeseries_for_station(s, nneighbours=9,
                                                                 start_date=start_date, end_date=end_date)
        if model_ts is not None:
            station_to_model_ts[s] = model_ts

    plot_utils.apply_plot_params(width_pt=None, height_cm=40.0, width_cm=30.0, font_size=10)
    fig = plt.figure()
    #two columns
    ncols = 3
    nrows = len(station_to_model_ts) / ncols
    if nrows * ncols < len(station_to_model_ts):
        nrows += 1
    gs = GridSpec(nrows, ncols, hspace=0.8, wspace=0.8)
    line_model, line_obs = None, None
    stations.sort(key=lambda x: x.latitude, reverse=True)

    i = -1

    ns_list = []
    station_list = []
    flow_acc_area_list = []

    for s, model_ts in station_to_model_ts.iteritems():
        i += 1
        ax = fig.add_subplot(gs[i // ncols, i % ncols])

        assert isinstance(model_ts, TimeSeries)

        #[t, m_data] = model_ts.get_daily_normals()
        #[t, s_data] = s.get_daily_normals()

        assert isinstance(s, Station)

        #climatologies
        #line_model = ax.plot(t, m_data, label = "Model (CRCM5)", lw = 3, color = "b")
        #line_obs = ax.plot(t, s_data, label = "Observation", lw = 3, color = "r")

        model_ts = model_ts.get_ts_of_daily_means()
        print model_ts.time[0], model_ts.time[-1]
        print model_ts.data[0:10]

        print model_ts.metadata

        mod_vals = model_ts.get_data_for_dates(s.dates)
        print mod_vals[:20]
        print "+" * 20
        assert len(mod_vals) == len(s.dates)

        mod_clims = []
        sta_clims = []
        for month in range(1, 13):
            bool_vector = np.array(map(lambda t: t.month == month, s.dates), dtype=np.bool)
            mod_clims.append(mod_vals[bool_vector].mean())
            sta_clims.append(np.array(s.values)[bool_vector].mean())

        mod_clims = np.array(mod_clims)
        sta_clims = np.array(sta_clims)

        line_model = ax.plot(range(1, 13), mod_clims, label="Model (CRCM5)", lw=1, color="b")
        line_obs = ax.plot(range(1, 13), sta_clims, label="Observation", lw=3, color="r", alpha=0.5)

        ax.annotate("r = {0:.2f}".format(float(np.corrcoef([mod_clims, sta_clims])[0, 1])),
                    xy=(0.6, 0.8), xycoords="axes fraction", ha="right")
        ax.annotate("ns = {0:.2f}".format(scores.nash_sutcliffe(mod_clims, sta_clims)),
                    xy=(0.6, 0.9), xycoords="axes fraction", ha="right")

        dt = model_ts.time[1] - model_ts.time[0]
        dt_sec = dt.days * 24 * 60 * 60 + dt.seconds
        #ax.annotate( "{0:g}".format( sum(mod_vals) * dt_sec ) + " ${\\rm m^3}$", xy = (0.7,0.7), xycoords= "axes fraction", color = "b")
        #ax.annotate( "{0:g}".format( sum(s.values) * dt_sec) + " ${\\rm m^3}$", xy = (0.7,0.6), xycoords= "axes fraction", color = "r")



        ax.set_title("%s: da_diff=%.1f%%, d = %.1f" % (s.id, (-s.drainage_km2 +
                                                              model_ts.metadata[
                                                                  "acc_area_km2"]) / s.drainage_km2 * 100.0,
                                                       model_ts.metadata["distance_to_obs_km"] ))

        #ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
        #ax.xaxis.set_major_locator(YearLocator())
        assert isinstance(ax, Axes)
        #fig.autofmt_xdate()
        #ax.xaxis.tick_bottom().set_rotation(60)

    lines = (line_model, line_obs)
    labels = ("Model (CRCM5)", "Observation" )
    fig.legend(lines, labels)
    fig.savefig("performance_86x86_0.5deg_river_ice_1yrspnp_const_manning.png")


def compare_streamflow():
    #lake level controlled only by routing
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1"

    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1_lakes_off_high_res"
    #coord_file = os.path.join(data_path, "pm1985050100_00000000p")


    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_198501_198612_0.1deg"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_88x88_0.5deg_with_lakes"

    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_with_lakes"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_88x88_0.5deg_with_lakes"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_without_lakes"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_lakes_flake"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_highres_spinup_12_month_without_lakes_v3"

    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3_sturm_snc"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_lakes_and_lakerof"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3_old_snc"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types"
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types_crcm_lk_fractions"
    coord_file = os.path.join(data_path, "pm1985050100_00000000p")

    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
                                    file_name_prefix="pm", all_files_in_samples_folder=True
    )
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                    "081006", "040830"]

    start_date = datetime(1986, 1, 1)
    end_date = datetime(1990, 12, 31)

    selected_ids = None
    stations = cehq_station.read_station_data(selected_ids=selected_ids,
                                              start_date=start_date, end_date=end_date
    )

    station_to_model_ts = {}
    for s in stations:
        assert isinstance(s, Station)

        if s.drainage_km2 <= 2 * np.radians(0.5) ** 2 * lat_lon.EARTH_RADIUS_METERS ** 2 * 1.0e-6:
            continue

        if not s.passes_rough_continuity_test(start_date, end_date):
            continue

        model_ts = manager.get_streamflow_timeseries_for_station(s, nneighbours=9,
                                                                 start_date=start_date, end_date=end_date)
        if model_ts is not None:
            station_to_model_ts[s] = model_ts

    plot_utils.apply_plot_params(width_pt=None, height_cm=40.0, width_cm=30.0, font_size=10)
    fig = plt.figure()
    #two columns
    ncols = 3
    nrows = len(station_to_model_ts) / ncols
    if nrows * ncols < len(station_to_model_ts):
        nrows += 1
    gs = GridSpec(nrows, ncols, hspace=0.8, wspace=0.8)
    line_model, line_obs = None, None
    stations.sort(key=lambda x: x.latitude, reverse=True)

    i = -1

    ns_list = []
    station_list = []
    flow_acc_area_list = []

    for s, model_ts in station_to_model_ts.iteritems():
        i += 1
        ax = fig.add_subplot(gs[i // ncols, i % ncols])

        assert isinstance(model_ts, TimeSeries)

        #[t, m_data] = model_ts.get_daily_normals()
        #[t, s_data] = s.get_daily_normals()

        assert isinstance(s, Station)

        #climatologies
        #line_model = ax.plot(t, m_data, label = "Model (CRCM5)", lw = 3, color = "b")
        #line_obs = ax.plot(t, s_data, label = "Observation", lw = 3, color = "r")

        model_ts = model_ts.get_ts_of_daily_means()
        print model_ts.time[0], model_ts.time[-1]
        print model_ts.data[0:10]

        print model_ts.metadata

        mod_vals = model_ts.get_data_for_dates(s.dates)
        print mod_vals[:20]
        print "+" * 20
        assert len(mod_vals) == len(s.dates)

        line_model = ax.plot(s.dates, mod_vals, label="Model (CRCM5)", lw=1, color="b")
        line_obs = ax.plot(s.dates, s.values, label="Observation", lw=3, color="r", alpha=0.5)

        ax.annotate("r = {0:.2f}".format(float(np.corrcoef([mod_vals, s.values])[0, 1])), xy=(0.7, 0.8),
                    xycoords="axes fraction")
        ax.annotate("ns = {0:.2f}".format(scores.nash_sutcliffe(mod_vals, s.values)), xy=(0.7, 0.9),
                    xycoords="axes fraction")

        dt = model_ts.time[1] - model_ts.time[0]
        dt_sec = dt.days * 24 * 60 * 60 + dt.seconds
        ax.annotate("{0:g}".format(sum(mod_vals) * dt_sec) + " ${\\rm m^3}$", xy=(0.7, 0.7), xycoords="axes fraction",
                    color="b")
        ax.annotate("{0:g}".format(sum(s.values) * dt_sec) + " ${\\rm m^3}$", xy=(0.7, 0.6), xycoords="axes fraction",
                    color="r")

        ax.set_title("%s: da_diff=%.1f%%, d = %.1f" % (s.id, (-s.drainage_km2 +
                                                              model_ts.metadata[
                                                                  "acc_area_km2"]) / s.drainage_km2 * 100.0,
                                                       model_ts.metadata["distance_to_obs_km"] ))

        ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
        ax.xaxis.set_major_locator(YearLocator())
        assert isinstance(ax, Axes)
        fig.autofmt_xdate()
        #ax.xaxis.tick_bottom().set_rotation(60)

    lines = (line_model, line_obs)
    labels = ("Model (CRCM5)", "Observation" )
    fig.legend(lines, labels)
    fig.savefig("performance_without_lakes_0.5deg_old_snc_with_lk_rof_crcm_lk_frct.png")

    pass


def plot_flow_directions_and_sel_stations(stations, dir_values):
    #TODO

    pass


def draw_drainage_area():
    plot_utils.apply_plot_params(width_pt=None, height_cm=20.0, width_cm=16, font_size=10)
    fig = plt.figure()

    manager = Crcm5ModelDataManager(
        samples_folder_path="data/from_guillimin/quebec_rivers_not_talk_with_lakes/Samples",
        file_name_prefix="physics"
    )

    basemap = Basemap(projection="omerc", no_rot=True, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0,
                      llcrnrlon=manager.lons2D[0, 0], llcrnrlat=manager.lats2D[0, 0],
                      urcrnrlon=manager.lons2D[-1, -1], urcrnrlat=manager.lats2D[-1, -1],
                      resolution="l"
    )

    x, y = basemap(manager.lons2D, manager.lats2D)
    acc_area = np.ma.masked_where(manager.flow_directions <= 0, manager.accumulation_area_km2)

    dx = x[1, 0] - x[0, 0]
    dy = y[0, 1] - y[0, 0]
    x -= dx / 2.0
    y -= dy / 2.0
    basemap.pcolormesh(x, y, acc_area)

    basemap.drawcoastlines()
    basemap.drawmeridians(np.arange(-180, 180, 20), labels=[0, 0, 0, 1])
    basemap.drawparallels(np.arange(-90, 90, 20), labels=[1, 0, 0, 0])
    plt.colorbar()

    plt.savefig("drainage_area.png")
    pass


def main():
    plot_utils.apply_plot_params(width_pt=None, height_cm=60, width_cm=20)
    #draw_drainage_area()
    #compare_streamflow_normals()
    compare_lake_levels()
    #
    # test_mean()
    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=20)
    #plot_utils.apply_plot_params(width_pt=None, height_cm=50, width_cm=50)
    #test_seasonal_mean()
    pass


def doTestRotPole():
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_lakes_flake"
    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
                                    file_name_prefix="pm", all_files_in_samples_folder=True)

    b = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(lons2d=manager.lons2D, lats2d=manager.lats2D)

    x, y = b(manager.lons2D, manager.lats2D)
    b.contourf(x, y, manager.mg)
    b.drawmeridians(np.arange(-180, 180, 30))
    b.drawparallels(np.arange(-90, 90, 30))

    b.drawcoastlines()

    ds = Dataset("/home/huziy/skynet1_rech3/Converters/NetCDF_converter/pm1958010100_00002232p_na_0.44deg_AU.nc")
    na_data = ds.variables["osr"][0, 0, :, :]
    rlons = ds.variables["rlon"][:]
    rlats = ds.variables["rlat"][:]

    rlats, rlons = np.meshgrid(rlats, rlons)

    lons = ds.variables["lon"][:]
    lats = ds.variables["lat"][:]

    b1 = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(
        lons2d=lons, lats2d=lats, lon_1=-97, lon_2=-7, lat_1=47.5, lat_2=0
    )
    x, y = b1(lons, lats)
    plt.figure()
    b1.contourf(x, y, na_data)
    b1.colorbar()
    b1.drawcoastlines()

    plt.show()


def doTestStuff():
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_lakes_flake"
    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
                                    file_name_prefix="pm", all_files_in_samples_folder=True)

    lf = manager.lake_fraction.transpose()
    lf = np.ma.masked_where(lf < 0.1, lf)

    plt.imshow(lf, origin="lower")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    import locale

    locale.setlocale(locale.LC_ALL, '')

    application_properties.set_current_directory()

    #testStuff()
    #doTestRotPole()
    main()
    print "Hello world"
