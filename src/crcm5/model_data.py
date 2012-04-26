from datetime import datetime, timedelta
import itertools
from matplotlib.dates import DateFormatter, DateLocator, MonthLocator
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap, maskoceans
from scipy.spatial.kdtree import KDTree
import application_properties
from data import cehq_station
from data.timeseries import DateValuePair, TimeSeries
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

        [distances, indices] = self.kdtree.query(x0, k = 8)

        acc_area_1d = self.accumulation_area_km2.flatten()

        da_diff = np.abs( acc_area_1d[indices] - station.drainage_km2) / station.drainage_km2

        max_dist = np.max(distances)
        obj_func = da_diff + distances / max_dist

        min_obj_func = np.min(obj_func)
        index_of_sel_index = np.where(min_obj_func == obj_func)
        select_index = indices[index_of_sel_index]

        #assert np.all(acc_area_1d >= 0)
        acc_area_1d[:select_index] = -1
        acc_area_1d[select_index+1:] = -1

        restored = np.reshape(acc_area_1d, self.accumulation_area_km2.shape)
        [i, j] = np.where(restored >= 0)

        ts = self._get_timeseries_for_point(i,j, start_date = start_date,
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


    def _get_timeseries_for_point(self, ix, iy,
                            start_date = None, end_date = None):
        """
        returns timeseries object for data: data[:, ix, iy]
        Note: uses caching in order to decrease IO operations
        """

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
                    hour_to_field = rpnObj.get_all_time_records_for_name(varname=self.var_name)
                    rpnObj.close()
                    print( hour_to_field.items()[0][0] , "for file {0}".format(the_path))
                    self.date_to_field.update(hour_to_field)


        for time, field in self.date_to_field.iteritems():
            if start_date is not None:
                if time < start_date: continue
            if end_date is not None:
                if time > end_date: continue

            value = field[ix, iy]
            dv.append(DateValuePair(date = time, value = value))

        dv.sort(key= lambda x: x.date)
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

    def get_timeseries_for_station(self, var_name = "", station = None):
        lon, lat = station.longitude, station.latitude
        x0 = lat_lon.lon_lat_to_cartesian(lon, lat)

        [distances, indices] = self.kdtree.query(x0, k = 1)

        self._get_timeseries_for_point()

        pass


def test_mean():
    plt.figure()
    manager = Crcm5ModelDataManager()


    #plt.pcolormesh(manager.get_monthly_mean_fields(months = [6])[6].transpose())

    basemap = Basemap(projection="omerc", no_rot=True, lon_1=-68, lat_1=52, lon_2=16.65, lat_2=0,
            llcrnrlon=manager.lons2D[0,0], llcrnrlat=manager.lats2D[0,0],
            urcrnrlon=manager.lons2D[-1,-1], urcrnrlat=manager.lats2D[-1, -1],
            resolution="l"
    )

    [x, y] = basemap(manager.lons2D, manager.lats2D)
    #data = manager.get_monthly_mean_fields(months=[6])[6]
    data = np.log( manager.accumulation_area_km2 )

    #get ocean mask
    #lons2D = manager.lons2D[:,:]
    #lons2D[lons2D >= 180] -= 360.0
    #ocean_mask = maskoceans(lons2D, manager.lats2D, data)

    data = np.ma.masked_where(manager.slope == -1, data)

    #data = np.ma.masked_where(data < 0.1, data)
    basemap.pcolormesh(x, y, data)
    plt.colorbar()
    basemap.drawcoastlines(linewidth=0.5)

    plt.savefig("mean.png")
    pass


def compare_lake_levels():
    manager = Crcm5ModelDataManager(samples_folder_path="data/from_guillimin/vary_lake_level1",
            file_name_prefix="pm", all_files_in_samples_folder=True
    )

    start_date = datetime(1985, 1, 1)
    end_date = datetime(1990, 12, 31)

    stations = cehq_station.read_station_data( folder="data/cehq_levels",
            start_date=start_date, end_date=end_date
    )

    plot_utils.apply_plot_params(width_pt=None, height_cm =30.0, width_cm=16, font_size=10)


    for s in stations:
        manager.get_timeseries_for_station(var_name = "")





def compare_streamflow():
    manager = Crcm5ModelDataManager(samples_folder_path="data/from_guillimin/vary_lake_level1",
            file_name_prefix="pm", all_files_in_samples_folder=True
    )
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                    "081006", "061502", "040830", "080718"]

    start_date = datetime(1985, 1, 1)
    end_date = datetime(1990, 12, 31)

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

        [t, m_data] = model_ts.get_daily_normals()


        [t, s_data] = s.get_daily_normals()

        assert len(s_data) == len(m_data)

        line_model = ax.plot(t, m_data, label = "Model (CRCM5)", lw = 3, color = "b")
        line_obs = ax.plot(t, s_data, label = "Observation", lw = 3, color = "r")

        ax.set_title("%s: drs=%.2f,drm=%.2f" % (s.id, s.drainage_km2,
                                                      model_ts.metadata["acc_area_km2"]))

        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(MonthLocator(bymonthday=15, bymonth=xrange(1,13,2)))

    lines = (line_model, line_obs)
    labels = ("Model (CRCM5)", "Observation" )
    fig.legend(lines, labels)
    fig.savefig("performance.png")

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
    #draw_drainage_area()
    compare_streamflow()
    #test_mean()
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  