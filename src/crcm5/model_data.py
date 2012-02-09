from datetime import datetime, timedelta
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
from rpn import RPN
import os

import matplotlib.pyplot as plt

class Crcm5ModelDataManager:

    def __init__(self, samples_folder_path = "data/gemclim/quebec/Samples",
                 var_name = "STFL", file_name_prefix = "pm"):
        self.samples_folder = samples_folder_path
        self._month_folder_prefix = None
        self.month_folder_name_format = "%s_%d%02d"
        self._read_lat_lon_fields()
        self._set_month_folder_prefix()
        self._read_static_data()
        self._set_simulation_start_date()
        self.var_name = var_name
        self.file_name_prefix = file_name_prefix
        self.date_to_field_cache = {}

        pass


    def get_timeseries_for_station(self, station,
                                    start_date = None, end_date = None
                                          ):
        """
        get model data for the gridcell corresponding to the station
        :type station: data.cehq_station.Station
        :rtype: data.timeseries.Timeseries
        """
        lon, lat = station.longitude, station.latitude
        x0 = lat_lon.lon_lat_to_cartesian_normalized(lon, lat)

        [distances, indices] = self.kdtree.query(x0, k = 32)

        acc_area_1d = self.accumulation_area_km2.flatten()

        da_diff = np.abs( acc_area_1d[indices] - station.drainage_km2)
        min_da_diff = min(da_diff)

        select_index = -1
        select_dist = np.inf
        for d, i, da_dist in zip(distances, indices, da_diff):
            if da_dist == min_da_diff:
                #if d < select_dist:
                select_dist = d
                select_index = i
        assert select_index >= 0
        acc_area_1d[:select_index] = -1
        acc_area_1d[select_index+1:] = -1

        restored = np.reshape(acc_area_1d, self.accumulation_area_km2.shape)
        [i, j] = np.where(restored > 0)

        ts = self._get_timeseries_for_point(i,j, start_date = start_date,
                        end_date = end_date)

        ts.metadata["acc_area_km2"] = acc_area_1d[select_index]
        return ts
        pass

    def _get_any_file_path(self):
        for month_folder in os.listdir(self.samples_folder):
            month_folder_path = os.path.join(self.samples_folder, month_folder)
            for the_file in os.listdir(month_folder_path):
                if the_file.startswith("."):
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
        [x, y, z] = lat_lon.lon_lat_to_cartesian_normalized(self.lons2D.flatten(), self.lats2D.flatten())
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


        month_folder_names = os.listdir(self.samples_folder)

        month_folder_paths = map( lambda x: os.path.join(self.samples_folder, x), month_folder_names)

        for month_folder_path in month_folder_paths:
            for file_name in os.listdir(month_folder_path):
                if file_name.startswith("."): continue
                if "_" not in file_name: continue
                if not file_name.startswith(self.file_name_prefix): continue

                file_path = os.path.join(month_folder_path, file_name)
                rpnObj = RPN(file_path)
                forecast_hour = rpnObj.get_current_validity_date()
                rpnObj.close()

                d = self.simulation_start_date + timedelta(hours = forecast_hour)
                if start_date is not None:
                    if d < start_date: continue

                if end_date is not None:
                    if d > end_date: continue

                if self.date_to_field_cache.has_key(d):
                    value = self.date_to_field_cache[d][ix, iy]
                else:
                    rpnObj = RPN(file_path)
                    field = rpnObj.get_first_record_for_name(self.var_name)
                    rpnObj.close()
                    self.date_to_field_cache[d] = field
                    value = field[ix, iy]

                dv.append(DateValuePair(date = d, value = value))


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
                    field_list.append( self._get_field_from_file(path_to_file=file_path, field_name=var_name) )

                #store mean field for the given month and year
                result[the_month].append(np.mean(field_list, axis=0))

        for m in months:
            result[m] = np.mean(result[m], axis=0)

        return result
        pass

    def _get_field_from_file(self, path_to_file = "",
                             field_name = "STFL"):
        """
        Read 2D data field from a file
        """
        rpnObj = RPN(path=path_to_file)
        data = rpnObj.get_first_record_for_name(field_name)
        rpnObj.close()
        return data



    def _read_static_data(self):
        """
         get drainage area fields

        """
        #TODO: change the way how the drainage area is read
        #TODO: i.e. instead of taking the margins just add the drainage area as the variable in the model

        file_path = os.path.join(self.samples_folder, "..")
        file_path = os.path.join(file_path, "infocell.rpn")
        rpnObj = RPN(file_path)
        #TODO: Custom margins, fix it
        self.accumulation_area_km2 = rpnObj.get_first_record_for_name("FACC")[10:-10, 10:-10]
        self.slope = rpnObj.get_first_record_for_name("SLOP")[10:-10, 10:-10]
        rpnObj.close()

        pass

    def _set_simulation_start_date(self):
        """
        Determine the starting date of the simulation
        """
        path = self._get_any_file_path()
        date_str = os.path.basename(path).split("_")[0][2:]
        self.simulation_start_date = datetime.strptime(date_str, "%Y%m%d%H")


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

def compare():
    manager = Crcm5ModelDataManager()
    selected_ids = ["104001", "103715", "093806", "093801", "092715",
                    "081006", "061502", "040830", "080718"]

    start_date = datetime(1985, 1, 1)
    end_date = datetime(1990, 12, 31)

    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )

    plot_utils.apply_plot_params(width_pt=None, height_cm =30.0, width_cm=16, font_size=12)
    fig = plt.figure()
    #two columns
    gs = GridSpec( len(stations) // 2 + len(stations) % 2, 2, hspace=0.4, wspace=0.4 )
    line_model, line_obs = None, None
    #: :type s: data.cehq_station.Station
    for i, s in enumerate(stations):
        model_ts = manager.get_timeseries_for_station(s, start_date = start_date, end_date = end_date)
        ax = fig.add_subplot( gs[i // 2, i % 2] )

        m_data = model_ts.get_monthly_normals()

        s_data = s.get_monthly_normals()

        line_model = ax.plot(xrange(1, 13), m_data, label = "Model (CRCM5)", lw = 3, color = "b")
        line_obs = ax.plot(xrange(1, 13), s_data, label = "Observation", lw = 3, color = "r")

        ax.set_title("%s: ns=%d, nm=%d,\n drs=%.2f,drm=%.2f" % (s.id, s.get_timeseries_length(),
                                                            model_ts.get_size(), s.drainage_km2,
                                                            model_ts.metadata["acc_area_km2"]))


    lines = (line_model, line_obs)
    labels = ("Model (CRCM5)", "Observation" )
    fig.legend(lines, labels)
    fig.savefig("performance.png")

    pass


def main():
    #TODO: implement
    #compare()
    test_mean()
    pass

if __name__ == "__main__":
    application_properties.set_current_directory()
    main()
    print "Hello world"
  