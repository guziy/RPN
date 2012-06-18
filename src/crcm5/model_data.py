from datetime import datetime
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
from data.timeseries import DateValuePair, TimeSeries
from permafrost import draw_regions
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
from rpn.rpn import RPN
import os

import matplotlib.pyplot as plt

class Crcm5ModelDataManager:

    def __init__(self, samples_folder_path = "data/gemclim/quebec/Samples",
                 var_name = "STFL", file_name_prefix = "pm",
                 all_files_in_samples_folder = False):
        self.file_name_prefix = file_name_prefix
        self.samples_folder = samples_folder_path
        self.all_files_in_one_folder = all_files_in_samples_folder
        self._read_lat_lon_fields()


        if not all_files_in_samples_folder:
            self._month_folder_prefix = None
            self.month_folder_name_format = "%s_%d%02d"
            self._set_month_folder_prefix()

        self._read_static_data()
        self.var_name = var_name

        self._file_paths = None

        self.date_to_field = {}
        self._flat_index_to_2d_cache = {}
        pass


    def _flat_index_to_2d(self, flat_index):
        if not len(self._flat_index_to_2d_cache):
            nx, ny = self.lons2D.shape
            j2d, i2d = meshgrid(xrange(ny), xrange(nx))
            i_flat = i2d.flatten()
            j_flat = j2d.flatten()
            self._flat_index_to_2d_cache = dict(zip(xrange(len(i_flat)), zip(i_flat,j_flat)))

        return self._flat_index_to_2d_cache[flat_index]



    def get_streamflow_timeseries_for_station(self, station, start_date = None, end_date = None, d_max_km = 20.0):
        """
        get timeseries but find model point for comparison using the following objectie function
        :type station: data.cehq_station.Station
        :rtype: data.timeseries.Timeseries
        """



        pass


    def get_streamflow_timeseries_for_station(self, station,
                                    start_date = None, end_date = None
                                          ):
        """
        get model data for the gridcell corresponding to the station
        :type station: data.cehq_station.Station
        :rtype: data.timeseries.Timeseries
        """
        lon, lat = station.longitude, station.latitude
        x0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        [distances, indices] = self.kdtree.query(x0, k = 20)

        acc_area_1d = self.accumulation_area_km2.flatten()

        da_diff = np.abs( acc_area_1d[indices] - station.drainage_km2) / station.drainage_km2

        max_dist = np.max(distances)
        obj_func = 8 * da_diff / np.max(da_diff) + distances / max_dist

        min_obj_func = np.min(obj_func)
        index_of_sel_index = np.where(min_obj_func == obj_func)
        select_index = indices[index_of_sel_index]

        #assert np.all(acc_area_1d >= 0)
        acc_area_1d[:select_index] = -1
        acc_area_1d[select_index+1:] = -1

        restored = np.reshape(acc_area_1d, self.accumulation_area_km2.shape)
        [i, j] = np.where(restored >= 0)

        ts = self.get_timeseries_for_point(i,j, start_date = start_date,
                        end_date = end_date)

        ts.metadata["acc_area_km2"] = acc_area_1d[select_index]
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

        if not len(self.date_to_field):
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
                    rpnObj = RPN(the_path)
                    hour_to_field = rpnObj.get_all_time_records_for_name(varname=var_name)
                    rpnObj.close()
                    print( hour_to_field.items()[0][0] , "for file {0}".format(the_path))
                    self.date_to_field.update(hour_to_field)


        for time, field in self.date_to_field.iteritems():
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

    def get_timeseries_for_station(self, var_name = "", station = None, nneighbours = 1, start_date = None, end_date = None):
        """
        :rtype:  TimeSeries
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




def compare_streamflow():
    #lake level controlled only by routing
    data_path = "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1_lakes_off"
    coord_file = os.path.join(data_path, "pm1985050100_00000000p")

    manager = Crcm5ModelDataManager(samples_folder_path=data_path,
            file_name_prefix="pm", all_files_in_samples_folder=True
    )
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                    "081006", "061502", "040830", "080718"]

    start_date = datetime(1986, 1, 1)
    end_date = datetime(1989, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )

    plot_utils.apply_plot_params(width_pt=None, height_cm =30.0, width_cm=16, font_size=10)
    fig = plt.figure()
    #two columns
    gs = GridSpec( len(stations) // 2 + len(stations) % 2, 2, hspace=0.4, wspace=0.4 )
    line_model, line_obs = None, None

    stations.sort(key=lambda x: x.latitude, reverse=True)


    for i, s in enumerate(stations):
        model_ts = manager.get_streamflow_timeseries_for_station(s, start_date = start_date, end_date = end_date)
        ax = fig.add_subplot( gs[i // 2, i % 2] )



        assert isinstance(model_ts, TimeSeries)






        #[t, m_data] = model_ts.get_daily_normals()
        #[t, s_data] = s.get_daily_normals()

        assert isinstance(s, Station)

        #climatologies
        #line_model = ax.plot(t, m_data, label = "Model (CRCM5)", lw = 3, color = "b")
        #line_obs = ax.plot(t, s_data, label = "Observation", lw = 3, color = "r")


        mod_vals = model_ts.get_data_for_dates(s.dates)
        line_model = ax.plot(s.dates, mod_vals, label = "Model (CRCM5)", lw = 1, color = "b")
        line_obs = ax.plot(s.dates, s.values, label = "Observation", lw = 3, color = "r", alpha = 0.5)

        ax.annotate( "r = {0:.2f}".format( float( np.corrcoef([mod_vals, s.values])[0,1] )), xy = (0.7,0.8), xycoords= "axes fraction")


        ax.set_title("%s: da_diff=%.2f %%" % (s.id, (-s.drainage_km2+
                                                      model_ts.metadata["acc_area_km2"]) / s.drainage_km2 * 100.0))

        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.xaxis.set_major_locator(YearLocator())

    lines = (line_model, line_obs)
    labels = ("Model (CRCM5)", "Observation" )
    fig.legend(lines, labels)
    fig.savefig("performance_without_lakes.png")

    pass



def draw_drainage_area():
    plot_utils.apply_plot_params(width_pt=None, height_cm =20.0, width_cm=16, font_size=10)
    fig = plt.figure()

    manager = Crcm5ModelDataManager(samples_folder_path="data/from_guillimin/quebec_rivers_not_talk_with_lakes/Samples",
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
    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=30)
    #draw_drainage_area()
    compare_streamflow()
    #compare_lake_levels()
    #
    # test_mean()
    plot_utils.apply_plot_params(width_pt=None, height_cm=20, width_cm=20)
    #plot_utils.apply_plot_params(width_pt=None, height_cm=50, width_cm=50)
    #test_seasonal_mean()
    pass

if __name__ == "__main__":
    import locale
    locale.setlocale(locale.LC_ALL, '')

    application_properties.set_current_directory()
    main()
    print "Hello world"
  