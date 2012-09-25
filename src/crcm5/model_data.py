from datetime import datetime, timedelta
import itertools
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from numpy.lib.function_base import meshgrid
from scipy.spatial.kdtree import KDTree
import application_properties
from data import cehq_station
from data.cehq_station import Station
from data.cell_manager import CellManager
from data.timeseries import DateValuePair, TimeSeries
from domains.rotated_lat_lon import RotatedLatLon
from permafrost import draw_regions
from rpn import level_kinds
from util import plot_utils, scores
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from rpn.rpn import RPN
import os

import matplotlib.pyplot as plt

import shelve
import pandas

class Crcm5ModelDataManager:

    def __init__(self, samples_folder_path = "data/gemclim/quebec/Samples",
                 var_name = "STFL", file_name_prefix = "pm",
                 all_files_in_samples_folder = False, need_cell_manager = False):
        self.file_name_prefix = file_name_prefix
        self.samples_folder = samples_folder_path
        self.all_files_in_one_folder = all_files_in_samples_folder
        self.need_cell_manager = need_cell_manager
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



    def get_daily_means(self, var_name = None):
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




    def get_date_to_field_dict(self, var_name = None):
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
                self.lon2D_rot[i,j], self.lat2D_rot[i,j] = rot_lat_lon_proj.toProjectionXY(self.lons2D[i,j], self.lats2D[i,j])

        return self.lon2D_rot, self.lat2D_rot

    def get_spatial_integral_over_mask_of_dyn_field(self, mask, weights_2d, path_to_folder = "", var_name = "",
                                         level = -1, level_kind = level_kinds.ARBITRARY, file_prefix = "dm"):


        """
        returns a timeseries object
        """
        weights_1d = weights_2d[mask == 1]
        data = {}
        for fName in os.listdir(path_to_folder):
            if not fName.startswith(file_prefix): continue
            fPath = os.path.join(path_to_folder, fName)
            rObj = RPN(fPath)
            t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level, level_kind=level_kind)
            for t, field in t_to_field.iteritems():
                data[t] = sum( field[mask == 1] * weights_1d )
            rObj.close()

        times = sorted(data.keys())
        vals = np.array( map(lambda t: data[t], times) )
        return TimeSeries(time=times, data=vals)



    def get_dynamic_field_for_point(self,ix,jy, path_to_folder = "", var_name = "",
                                     level = -1, level_kind = level_kinds.ARBITRARY, file_prefix = "dm"):

        """
        returns a timeseries object
        """
        data = {}
        for fName in os.listdir(path_to_folder):
            if not fName.endswith(file_prefix): continue
            fPath = os.path.join(path_to_folder, fName)
            rObj = RPN(fPath)
            t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level, level_kind=level_kind)
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
            self._flat_index_to_2d_cache = dict(zip(xrange(len(i_flat)), zip(i_flat,j_flat)))

        return self._flat_index_to_2d_cache[flat_index]




    def get_monthly_sums_over_points(self, mask, var_name, level = -1 , level_kind = level_kinds.ARBITRARY,
         areas2d = None, start_date = None, end_date = None):
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
                if not os.path.basename(fPath).startswith( self.file_name_prefix): continue
                rObj = RPN(fPath)
                t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level, level_kind=level_kind)
                for the_key in t_to_field.keys():
                    the_field = t_to_field[the_key]
                    data[the_key] = np.sum(the_field[mask == 1] * areas1d)
                rObj.close()
                pass


        dates = sorted(data.keys())
        dates = list( itertools.ifilter(lambda t: start_date <= t <= end_date, dates ) )
        dt = dates[1] - dates[0]
        return TimeSeries(data=map(lambda t: data[t], dates), time=dates).get_ts_of_monthly_integrals_in_time(), dt



    

    def get_monthly_means_over_points(self, mask, var_name, level = -1 , level_kind = level_kinds.ARBITRARY,
         areas2d = None, start_date = None, end_date = None):
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
                t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level, level_kind=level_kind)
                for the_key in t_to_field.keys():
                    the_field = t_to_field[the_key]
                    data[the_key] = np.sum(the_field[mask == 1] * areas1d)
                rObj.close()
                pass


        dates = sorted(data.keys())
        dates = list( itertools.ifilter(lambda t: start_date <= t <= end_date, dates ) )
        return TimeSeries(data=map(lambda t: data[t], dates), time=dates).get_ts_of_monthly_means()




    def get_daily_means_over_points(self, mask, var_name, level = -1 , level_kind = level_kinds.ARBITRARY,
         areas2d = None, start_date = None, end_date = None):
        """
        return Timeseries object with values = sum(data[ti][mask == 1] * Area[mask == 1]) / sum(Area[mask == 1])
        """
        day = timedelta(days = 1)
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
                t_to_field = rObj.get_all_time_records_for_name_and_level(varname=var_name, level=level, level_kind=level_kind)
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




    def _get_model_indices_for_stfl_station(self, station, nneighbours = 4):
        """
        Selects a model point corresponding to the streamflow measuring station
        """
        lon, lat = station.longitude, station.latitude
        x0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        [distances, indices] = self.kdtree.query(x0, k = nneighbours)

        acc_area_1d = self.accumulation_area_km2.flatten()

        da_diff = np.abs( acc_area_1d[indices] - station.drainage_km2) / station.drainage_km2

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
        acc_area_1d[select_index+1:] = -1

        restored = np.reshape(acc_area_1d, self.accumulation_area_km2.shape)
        [i, j] = np.where(restored >= 0)

        if len(i) > 1:
            print "Warning {0} candidate cells for the station {1}".format(len(i), station.id)
        return i[0],j[0]




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
                                    start_date = None, end_date = None,
                                    var_name = None, nneighbours = 4):
        """
        get model data for the gridcell corresponding to the station
        :type station: data.cehq_station.Station
        :rtype: data.timeseries.Timeseries
        """

        i, j = self._get_model_indices_for_stfl_station(station, nneighbours=nneighbours)
        if i < 0: return None
        print "retrieving ts for point ({0}, {1})".format(i, j)
        ts = self.get_timeseries_for_point(i[0],j[0], start_date = start_date,
                        end_date = end_date, var_name=var_name)

        ts.metadata = self._set_metadata(i, j)
        ts.metadata["distance_to_obs_km"] = 1.0e-3 * lat_lon.get_distance_in_meters(self.lons2D[i, j], self.lats2D[i,j],
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

        #create kdtree for easier and faster lookup of the corresponding points
        #model <=> obs, for comparison
        [x, y, z] = lat_lon.lon_lat_to_cartesian(self.lons2D.flatten(), self.lats2D.flatten())
        self.kdtree = KDTree(zip(x, y, z))


        pass




    def get_omerc_basemap(self, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0, resolution = "l"):
        basemap = Basemap(projection="omerc", no_rot=True, lon_1=lon_1, lat_1=lat_1,
                lon_2=lon_2, lat_2=lat_2,
                llcrnrlon=self.lons2D[0,0], llcrnrlat=self.lats2D[0,0],
                urcrnrlon=self.lons2D[-1,-1], urcrnrlat=self.lats2D[-1, -1],
                resolution=resolution
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
            all_years.append( int( file_name.split("_")[-1][:-2] ) )
        return min(all_years), max(all_years)





    def get_streamflow_dataframe_for_stations(self, station_list, start_date = None, end_date = None,
                                        var_name = None, nneighbours = 4,
                                        distance_upper_bound_m = np.Inf ):

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
            metadata["distance_to_obs_km"] = lat_lon.get_distance_in_meters(self.lons2D[i, j], self.lats2D[i,j],
                        s.longitude, s.latitude
            )
            station_to_cell_props[s.id] = metadata


        id_list = map(lambda s: s.id, station_list)
        df = self.get_timeseries_for_points(ix_list, jy_list, id_list , start_date=start_date, end_date=end_date, var_name=var_name)
        assert isinstance(df, pandas.DataFrame)
        print df
        return df, station_to_cell_props





    def get_timeseries_for_points(self, ix_list, jy_list, id_list,
                                start_date = None, end_date = None,
                                var_name = None):
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
            for the_path, fName in zip( paths, fNames ):
                if os.path.isfile(the_path):
                    print "Opening the file {0} ...".format(the_path)
                    rpnObj = RPN(the_path)
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
                            start_date = None, end_date = None,
                            var_name = None, weight = 1):
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
            dv.append(DateValuePair(date = time, value = value))

        dv.sort(key= lambda x: x.date)
        print "datei = {0}, datef = {1}".format(dv[0].date, dv[-1].date)
        return TimeSeries(data = map(lambda x: x.value, dv),
                          time = map(lambda x: x.date, dv) )

        pass



    def get_mean_field(self,start_year, end_year, months = None, file_name_prefix = "pm",
                       var_name = "STFL", level = -1, level_kind = level_kinds.ARBITRARY):
        if self.all_files_in_one_folder:
            fNames = itertools.ifilter( lambda theName: theName.startswith(file_name_prefix), os.listdir(self.samples_folder) )
            fNames = list(fNames)
            fPaths = map(lambda x: os.path.join(self.samples_folder, x), fNames)

            fields_list = []
            for fPath in fPaths:
                rObj = RPN(fPath)
                data = rObj.get_all_time_records_for_name_and_level(varname = var_name, level=level, level_kind=level_kind)
                for t, field in data.iteritems():
                    if start_year <= t.year <= end_year:
                        if t.month in months:
                            fields_list.append(field)
            return np.mean(fields_list, axis=0)
        else:
            raise NotImplementedError("Need to implement the case of this data organization, or put all the data files to the same folder.")

        pass

    def get_monthly_mean_fields(self, start_date = None,
                                      end_date = None,
                                      months = xrange(1,13),
                                      var_name = "STFL",
                                      file_name_prefix = "pm"
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
                    field_list.append( self._get_2d_field_from_file(path_to_file=file_path, field_name=var_name) )

                #store mean field for the given month and year
                result[the_month].append(np.mean(field_list, axis=0))

        for m in months:
            result[m] = np.mean(result[m], axis=0)

        return result
        pass

    def _get_2d_field_from_file(self, path_to_file = "",
                             field_name = "STFL"):
        """
        Read 2D data field from a file
        """
        rpnObj = RPN(path=path_to_file)
        data = rpnObj.get_first_record_for_name(field_name)
        rpnObj.close()
        return data



    def get_monthly_climatology(self, varname =  "STFL"):
        time_series = {}
        if self.all_files_in_one_folder:
            for fName in os.listdir(self.samples_folder):
                fPath = os.path.join(self.samples_folder, fName)
                rpnObj = RPN(fPath)
                date_to_field = rpnObj.get_all_time_records_for_name(varname=varname)
                time_series.update(date_to_field)

        result = []
        for month in xrange(1,13):
            sel_dates = itertools.ifilter(lambda x: x.month == month, time_series.keys())
            sel_values = map(lambda x: time_series[x], sel_dates)
            result.append(np.array(sel_values).mean(axis = 0))

        return result


    def get_mean_in_time(self, var_name = "STFL"):
        all_fields = []
        if self.all_files_in_one_folder:
            for fName in os.listdir(self.samples_folder):
                fPath = os.path.join(self.samples_folder, fName)
                rpnObj = RPN(fPath)
                all_fields.extend( rpnObj.get_all_time_records_for_name(varname = var_name).values())
        return np.mean(all_fields, axis=0)


    def _read_static_data(self, derive_from_data = True):
        """
         get drainage area fields

        """
        #TODO: change the way how the drainage area is read
        #TODO: i.e. instead of taking the margins just add the drainage area as the variable in the model

        if derive_from_data and self.all_files_in_one_folder:
            files = os.listdir(self.samples_folder)
            files = itertools.ifilter(lambda x: x.startswith(self.file_name_prefix)
                and not x.startswith("."), files
            )
            file = sorted(files)[0]
            file_path = os.path.join(self.samples_folder, file)
            rpnObj = RPN(file_path)
            self.accumulation_area_km2 = rpnObj.get_first_record_for_name("FAA")
            self.flow_directions = rpnObj.get_first_record_for_name("FLDR")
            self.lake_fraction = rpnObj.get_first_record_for_name("LF1")
            self.bankfull_storage_m3 = rpnObj.get_first_record_for_name("STBM")

            try:
                self.cell_area = rpnObj.get_first_record_for_name("DX")
            except Exception, e:
                print e.message
                print "Could not find cell area in the output: ", file_path

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







    def get_timeseries_for_station(self, var_name = "", station = None,
                                   nneighbours = 1, start_date = None, end_date = None):
        """
        :rtype: list of  TimeSeries  or None
        """
        lon, lat = station.longitude, station.latitude
        x0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        [distances, indices] = self.kdtree.query(x0, k = nneighbours)

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
        if self.cell_manager is not None:
            aCell = self.cell_manager.cells[i_model0][j_model0]
            return self.cell_manager.get_mask_of_cells_connected_with(aCell)
        else:
            print "CellManager should be supplied for this operation."
            raise Exception("Cellmanager should be created via option need_cell_manager = True in the constructor.")


def test_seasonal_mean():
    fig = plt.figure()
    assert isinstance(fig, Figure)
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1"
    manager = Crcm5ModelDataManager(samples_folder_path=data_path, all_files_in_samples_folder=True

    )


    gs = gridspec.GridSpec(4,3)

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
        d = datetime(2000,i+1,1)
        ax.set_title(d.strftime("%B"))

        data = np.ma.masked_all(the_mean.shape)
        data[cond] = (field[cond] - the_mean[cond]) / the_mean[cond] * 100.0
        #data = np.ma.masked_where( ~cond, data )
        print np.ma.min(data), np.ma.max(data)
        #data = np.ma.log(data)
        img = basemap.pcolormesh(x, y, data, ax = ax,
                    vmax = 100.0, vmin = -100, cmap = colormap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(img, ax = ax, cax = cax, extend = "both")
        basemap.drawcoastlines(linewidth=0.5, ax = ax)
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


def test_mean():
    plt.figure()
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1"
    manager = Crcm5ModelDataManager(samples_folder_path=data_path,all_files_in_samples_folder=True

    )


    #plt.pcolormesh(manager.get_monthly_mean_fields(months = [6])[6].transpose())

    basemap = Basemap(projection="omerc", no_rot=True, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0,
            llcrnrlon=manager.lons2D[0,0], llcrnrlat=manager.lats2D[0,0],
            urcrnrlon=manager.lons2D[-1,-1], urcrnrlat=manager.lats2D[-1, -1],
            resolution="l"
    )

    [x, y] = basemap(manager.lons2D, manager.lats2D)
    data = manager.get_mean_in_time(var_name="STFL")

    #get ocean mask
    #lons2D = manager.lons2D[:,:]
    #lons2D[lons2D >= 180] -= 360.0
    #ocean_mask = maskoceans(lons2D, manager.lats2D, data)

    data = np.ma.masked_where( (data < 50), data)

    #data = np.ma.masked_where(data < 0.1, data)
    basemap.pcolormesh(x, y, data)
    plt.colorbar()
    basemap.drawcoastlines(linewidth=0.5)

    plt.savefig("mean.png")
    pass


def compare_lake_levels():
    #lake level controlled only with evaporation and precipitation
    data_path = "data/from_guillimin/vary_lake_level1"

    #lake level controlled only by routing
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1"

    selected_ids = [
        "093807", "011508", "061303", "061304", "040408", "030247"
    ]

    coord_file = os.path.join(data_path, "pm1985010100_00000000p")


    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
            file_name_prefix="pm", all_files_in_samples_folder=True, var_name="CLDP"
    )

    start_date = datetime(1987, 1, 1)
    end_date = datetime(1987, 12, 31)

    stations = cehq_station.read_station_data( folder="data/cehq_levels",
            start_date=start_date, end_date=end_date, selected_ids=selected_ids
    )

    plot_utils.apply_plot_params(width_pt=None, height_cm =30.0, width_cm=16, font_size=10)


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
        mod_ts_all = manager.get_timeseries_for_station(var_name = "CLDP", station=s, nneighbours=1,
            start_date = start_date, end_date= end_date
        )
        if mod_ts_all is None: #means model does not see lakes in the vicinity
            continue
        print(s.id,":",s.get_timeseries_length())

        mod_normals_all = []



        for mod_ts in mod_ts_all:
            mod_day_dates, mod_normals = mod_ts.get_daily_normals(start_date=start_date, end_date=end_date, stamp_year=2001)
            mod_normals_all.append(mod_normals)
        mod_normals = np.mean(mod_normals_all, axis = 0)

        if np.max(mod_normals) < 0.1: continue

        sta_day_dates, sta_normals = s.get_daily_normals(start_date=start_date, end_date=end_date, stamp_year=2001)

        if None in [ sta_day_dates, sta_normals ]:
            continue

        r = sub_p // ncols
        c = sub_p % ncols
        sub_p += 1
        ax = fig.add_subplot(gs[r, c])
        assert isinstance(ax, Axes)
        print len(mod_normals), len(sta_day_dates)
        #normals
        #h_m = ax.plot(sta_day_dates, mod_normals - np.mean(mod_normals) , "b", label = "model", lw = 3)
        #h_s = ax.plot(sta_day_dates, sta_normals - np.mean(sta_normals), "r", label = "station", lw = 3)

        #instantaneous values
        h_m = ax.plot(mod_ts_all[0].time, mod_ts_all[0].data - np.mean(mod_ts_all[0].data) , "b", label = "model", lw = 3)
        h_s = ax.plot(s.dates, s.values - np.mean(s.values), "r", label = "station", lw = 1)



        ax.set_title(s.id + "({0:.3f}, {1:.3f})".format(s.longitude, s.latitude))
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=15, interval=2))


        ax.yaxis.set_major_locator(LinearLocator(numticks=5))

        selected_stations.append(s)
        #ax.legend()

    ax = fig.add_subplot(gs[(r+1):,:])
    b, lons2d, lats2d = draw_regions.get_basemap_and_coords(
        file_path=coord_file,
        lon1=-68, lat1=52, lon2=16.65, lat2=0
    )
    b.drawcoastlines()
    y1 = 0.8
    dy = y1 / float( len(selected_stations) )

    x1 = [0.05, 0.85]
    for i, s in enumerate(selected_stations):
        x, y = b(s.longitude, s.latitude)
        b.scatter(x,y,c="r",marker="^", zorder = 5, s = 100)
        xy_text = x1[i%2], y1
        y1 -= dy
        ax.annotate(s.id, xy=(x, y), xytext = xy_text, textcoords = "axes fraction",
             bbox = dict(facecolor = 'white'), weight = "bold",
            arrowprops=dict(facecolor='black', arrowstyle="->")
        )

    fig.legend([h_m, h_s], ["Model", "Obs."], "lower right")
    fig.tight_layout(h_pad=2)
    fig.savefig("lake_level_comp_mean_anomalies.png")


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
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3_old_snc"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types"
    #data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types_crcm_lk_fractions"
    coord_file = os.path.join(data_path, "pm1985050100_00000000p")


    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
            file_name_prefix="pm", all_files_in_samples_folder=True
    )
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                    "081006", "040830"]

    start_date = datetime(1986, 1, 1)
    end_date = datetime(1990, 12, 31)

    selected_ids = None
    stations = cehq_station.read_station_data(selected_ids = selected_ids,
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

        model_ts = manager.get_streamflow_timeseries_for_station(s,nneighbours=9,
            start_date = start_date, end_date = end_date)
        if model_ts is not None:
            station_to_model_ts[s] = model_ts

    plot_utils.apply_plot_params(width_pt=None, height_cm =40.0, width_cm=30.0, font_size=10)
    fig = plt.figure()
    #two columns
    ncols = 3
    nrows = len(station_to_model_ts) / ncols
    if nrows * ncols < len(station_to_model_ts):
        nrows += 1
    gs = GridSpec( nrows, ncols, hspace=0.8, wspace=0.8 )
    line_model, line_obs = None, None
    stations.sort(key=lambda x: x.latitude, reverse=True)

    i = -1

    ns_list = []
    station_list = []
    flow_acc_area_list = []

    for s, model_ts in station_to_model_ts.iteritems():
        i += 1
        ax = fig.add_subplot( gs[i // ncols , i % ncols] )

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
        for month in range(1,13):
            bool_vector = np.array( map(lambda t: t.month == month, s.dates), dtype=np.bool )
            mod_clims.append(mod_vals[bool_vector].mean())
            sta_clims.append(np.array(s.values)[bool_vector].mean())

        mod_clims = np.array(mod_clims)
        sta_clims = np.array(sta_clims)


        line_model = ax.plot(range(1,13), mod_clims, label = "Model (CRCM5)", lw = 1, color = "b")
        line_obs = ax.plot(range(1,13), sta_clims, label = "Observation", lw = 3, color = "r", alpha = 0.5)

        ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod_clims, sta_clims])[0,1] )),
            xy = (0.6,0.8), xycoords= "axes fraction", ha = "right")
        ax.annotate("ns = {0:.2f}".format(scores.nash_sutcliffe(mod_clims, sta_clims)),
            xy = (0.6,0.9), xycoords= "axes fraction", ha = "right")

        dt = model_ts.time[1] - model_ts.time[0]
        dt_sec = dt.days * 24 * 60 * 60 + dt.seconds
        #ax.annotate( "{0:g}".format( sum(mod_vals) * dt_sec ) + " ${\\rm m^3}$", xy = (0.7,0.7), xycoords= "axes fraction", color = "b")
        #ax.annotate( "{0:g}".format( sum(s.values) * dt_sec) + " ${\\rm m^3}$", xy = (0.7,0.6), xycoords= "axes fraction", color = "r")



        ax.set_title("%s: da_diff=%.1f%%, d = %.1f" % (s.id, (-s.drainage_km2+
                        model_ts.metadata["acc_area_km2"]) / s.drainage_km2 * 100.0,
                        model_ts.metadata["distance_to_obs_km"] ))

        #ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
        #ax.xaxis.set_major_locator(YearLocator())
        assert isinstance(ax, Axes)
        #fig.autofmt_xdate()
        #ax.xaxis.tick_bottom().set_rotation(60)


    lines = (line_model, line_obs)
    labels = ("Model (CRCM5)", "Observation" )
    fig.legend(lines, labels)
    fig.savefig("performance_clim_without_lakes.png")


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
    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )





    station_to_model_ts = {}
    for s in stations:
        assert isinstance(s, Station)


        if s.drainage_km2 <= 2 * np.radians(0.5) ** 2 * lat_lon.EARTH_RADIUS_METERS ** 2 * 1.0e-6:
            continue

        if not s.passes_rough_continuity_test(start_date, end_date):
           continue

        model_ts = manager.get_streamflow_timeseries_for_station(s,nneighbours=9,
            start_date = start_date, end_date = end_date)
        if model_ts is not None:
            station_to_model_ts[s] = model_ts

    plot_utils.apply_plot_params(width_pt=None, height_cm =40.0, width_cm=30.0, font_size=10)
    fig = plt.figure()
    #two columns
    ncols = 3
    nrows = len(station_to_model_ts) / ncols
    if nrows * ncols < len(station_to_model_ts):
        nrows += 1
    gs = GridSpec( nrows, ncols, hspace=0.8, wspace=0.8 )
    line_model, line_obs = None, None
    stations.sort(key=lambda x: x.latitude, reverse=True)

    i = -1

    ns_list = []
    station_list = []
    flow_acc_area_list = []

    for s, model_ts in station_to_model_ts.iteritems():
        i += 1
        ax = fig.add_subplot( gs[i // ncols , i % ncols] )

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

        line_model = ax.plot(s.dates, mod_vals, label = "Model (CRCM5)", lw = 1, color = "b")
        line_obs = ax.plot(s.dates, s.values, label = "Observation", lw = 3, color = "r", alpha = 0.5)

        ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod_vals, s.values])[0,1] )), xy = (0.7,0.8), xycoords= "axes fraction")
        ax.annotate("ns = {0:.2f}".format(scores.nash_sutcliffe(mod_vals, s.values)), xy = (0.7,0.9), xycoords= "axes fraction")

        dt = model_ts.time[1] - model_ts.time[0]
        dt_sec = dt.days * 24 * 60 * 60 + dt.seconds
        ax.annotate( "{0:g}".format( sum(mod_vals) * dt_sec ) + " ${\\rm m^3}$", xy = (0.7,0.7), xycoords= "axes fraction", color = "b")
        ax.annotate( "{0:g}".format( sum(s.values) * dt_sec) + " ${\\rm m^3}$", xy = (0.7,0.6), xycoords= "axes fraction", color = "r")



        ax.set_title("%s: da_diff=%.1f%%, d = %.1f" % (s.id, (-s.drainage_km2+
                        model_ts.metadata["acc_area_km2"]) / s.drainage_km2 * 100.0,
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
    plot_utils.apply_plot_params(width_pt=None, height_cm =20.0, width_cm=16, font_size=10)
    fig = plt.figure()

    manager = Crcm5ModelDataManager(
        samples_folder_path="data/from_guillimin/quebec_rivers_not_talk_with_lakes/Samples",
        file_name_prefix="physics"
    )


    basemap = Basemap(projection="omerc", no_rot=True, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0,
            llcrnrlon=manager.lons2D[0,0], llcrnrlat=manager.lats2D[0,0],
            urcrnrlon=manager.lons2D[-1,-1], urcrnrlat=manager.lats2D[-1, -1],
            resolution="l"
    )

    x, y = basemap( manager.lons2D, manager.lats2D )
    acc_area = np.ma.masked_where(manager.flow_directions <= 0, manager.accumulation_area_km2)

    dx = x[1,0] - x[0,0]
    dy = y[0,1] - y[0,0]
    x -= dx / 2.0
    y -= dy / 2.0
    basemap.pcolormesh(x, y, acc_area)

    basemap.drawcoastlines()
    basemap.drawmeridians(np.arange(-180, 180, 20), labels=[0,0,0,1])
    basemap.drawparallels(np.arange(-90, 90, 20), labels=[1,0,0,0])
    plt.colorbar()

    plt.savefig("drainage_area.png")
    pass


def main():
    plot_utils.apply_plot_params(width_pt=None, height_cm=60, width_cm=20)
    #draw_drainage_area()
    compare_streamflow_normals()
    #compare_lake_levels()
    #
    # test_mean()
    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=20)
    #plot_utils.apply_plot_params(width_pt=None, height_cm=50, width_cm=50)
    #test_seasonal_mean()
    pass


def testStuff():
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

    main()
    print "Hello world"
  
