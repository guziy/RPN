from datetime import datetime, timedelta

__author__ = 'huziy'

import time
import itertools
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, DayLocator
from crcm5.model_data import Crcm5ModelDataManager

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import tables as tb


def get_basemap_from_hdf(file_path=""):
    """
    :param file_path:
    :return: lons(2d), lats(2), basemap - corresponding to the data in the file
    """
    h = tb.open_file(file_path)

    # Extract 2d longitudes and latitudes
    lons = h.getNode("/", "longitude")[:]
    lats = h.getNode("/", "latitude")[:]

    rotpoletable = h.getNode("/", "rotpole")

    assert isinstance(rotpoletable, tb.Table)

    params = {}
    for row in rotpoletable:
        params[row["name"]] = row["value"]
    rotpoletable.close()

    basemap = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(
        lons2d=lons, lats2d=lats,
        lon_1=params["lon1"], lon_2=params["lon2"],
        lat_1=params["lat1"], lat_2=params["lat2"]
    )
    h.close()
    return lons, lats, basemap


def get_array_from_file(path="", var_name=""):
    h = tb.open_file(path)

    if not var_name in h.root:
        print "Warning: no {0} in {1}".format(var_name, path)
        return None

    data = h.get_node("/", var_name)[:]
    h.close()
    return data


def get_mean_2d_fields_for_months(path="", var_name="", level=None, months=None,
                                  start_year=None,
                                  end_year=None):
    """
    Return means over the specified months for each year
    :param path: path to the hdf file with data
    :param var_name:
    :param level:
    """
    return Crcm5ModelDataManager.hdf_get_seasonal_means(path_to_hdf=path, months=months, var_name=var_name, level=level,
                                                        start_year=start_year, end_year=end_year)




def get_lake_level_timesries_due_to_precip_evap(path="", i_index=None, j_index=None):
    """

    :param path:
    :param i_index:
    :param j_index:
    :return:
    """
    h = tb.open_file(path)
    traf_table = h.get_node("/", "TRAF")
    sel_rows = traf_table.where("level == 6")
    dates = []
    vals = []
    for the_row in sel_rows:
        dates.append(datetime(the_row["year"], the_row["month"], the_row["day"], the_row["hour"]))
        vals.append(the_row["field"][i_index, j_index])

    ts = pd.TimeSeries(data=vals, index=dates)
    ts = ts.sort_index()

    dt = ts.index[1] - ts.index[0]
    print "dt = ", dt
    ts_cldp = ts.cumsum() * dt.total_seconds()
    return ts_cldp


def get_daily_climatology_for_a_point_cldp_due_to_precip_evap(path="", i_index=None, j_index=None,
                                                              year_list = None):
    """

    :param path:
    :param i_index:
    :param j_index:
    :return:
    """
    ts = get_lake_level_timesries_due_to_precip_evap(path=path, i_index=i_index, j_index=j_index)


    assert isinstance(ts, pd.TimeSeries)
    ts = ts.select(lambda d: d.year in year_list)


    ts_clim = ts.groupby(
        lambda d: datetime(2001, d.month, d.day, 1) if not (d.month == 2 and d.day == 29) else
        datetime(2001, d.month, d.day - 1, 1)).mean()

    print type(ts_clim)
    print dir(ts_clim)

    assert isinstance(ts_clim, pd.Series)
    ts_clim = ts_clim.sort_index()

    #assert isinstance(ts_clim, pd.TimeSeries)
    print ts_clim.index
    return ts_clim.index, ts_clim.values


def get_daily_climatology_for_a_point(path="", var_name="STFL", level=None,
                                      years_of_interest=None,
                                      i_index=None, j_index=None):
    """

    :rtype : tuple
    :param years_of_interest: is a list of years used for calculating daily climatologies
    """
    h = tb.open_file(path)

    var_table = h.get_node("/", var_name)

    def grouping_func(the_row):
        return the_row["month"], the_row["day"], the_row["level"]

    date_to_mean = {}
    date_to_count = {}

    if level is not None:
        selbylevel = "(level == {0})".format(level)
    else:
        selbylevel = ""

    selbyyear = "|".join(["(year == {0})".format(y) for y in years_of_interest])

    if level is not None:
        sel_for_years_and_level = var_table.where(
            "({0}) & ({1}) & ~((month == 2) & (day == 29))".format(selbylevel, selbyyear))
    else:
        print "({0}) & ((month != 2) | (day != 29))".format(selbyyear)
        sel_for_years_and_level = var_table.where(
            "({0}) & ((month != 2) | (day != 29))".format(selbyyear))

    for gKey, selrows in itertools.groupby(sel_for_years_and_level, grouping_func):
        the_month, the_day, the_level = gKey
        d = datetime(2001, the_month, the_day)
        n0 = 0
        x0 = 0
        if d in date_to_count:
            n0 = date_to_count[d]
            x0 = date_to_mean[d]

        x1 = np.asarray([row["field"][i_index, j_index] for row in selrows])
        n1 = x1.shape[0]
        x1 = x1.mean(axis=0)
        x1 = (n1 * x1 + n0 * x0) / float(n0 + n1)
        date_to_mean[d] = x1
        date_to_count[d] = n0 + n1

    h.close()
    sorted_dates = list(sorted(date_to_mean.keys()))
    return sorted_dates, [date_to_mean[d] for d in sorted_dates]


def get_daily_climatology(path_to_hdf_file="", var_name="STFL", level=None, start_year=None, end_year=None):
    if var_name.endswith("_min"):
        return get_daily_min_climatology(path_to_hdf_file=path_to_hdf_file, var_name=var_name, level=level,
                                         start_year=start_year, end_year=end_year)
    elif var_name.endswith("_max"):
        return get_daily_max_climatology(path_to_hdf_file=path_to_hdf_file, var_name=var_name, level=level,
                                         start_year=start_year, end_year=end_year)
    else:
        return Crcm5ModelDataManager.hdf_get_daily_climatological_fields(hdf_db_path=path_to_hdf_file,
                                                                         start_year=start_year,
                                                                         end_year=end_year,
                                                                         var_name=var_name,
                                                                         level=level,
                                                                         use_grouping=True)


def get_daily_max_climatology(path_to_hdf_file="", var_name="STFL", level=None,
                              start_year=None, end_year=None):
    var_name_max = "{0}_max".format(var_name)

    return Crcm5ModelDataManager.hdf_get_daily_extreme_climatological_fields(
        hdf_db_path=path_to_hdf_file,
        start_year=start_year, end_year=end_year, var_name=var_name_max,
        level=level, maximum=True)


def get_daily_min_climatology(path_to_hdf_file="", var_name="STFL", level=None,
                              start_year=None, end_year=None):
    var_name_min = "{0}".format(var_name)

    return Crcm5ModelDataManager.hdf_get_daily_extreme_climatological_fields(
        hdf_db_path=path_to_hdf_file,
        start_year=start_year, end_year=end_year, var_name=var_name_min,
        level=level, maximum=False)


def get_daily_climatology_of_3d_field(path_to_hdf_file="", var_name="STFL", start_year=None,
                                      end_year=None):
    """

    :param path_to_hdf_file:
    :param var_name:
    :param start_year:
    :param end_year:
    :return: sorted_dates, sorted_levels, data (t, lev, x, y)
    """
    h = tb.open_file(path_to_hdf_file, "a")

    clim_3d_node = "/daily_climatology_3d"
    data_node_path = "{0}/{1}".format(clim_3d_node, var_name)
    levels_node_path = "{0}/{1}_levels".format(clim_3d_node, var_name)
    if data_node_path in h:
        var_arr = h.get_node(data_node_path)
        d0 = datetime(2001, 1, 1)
        dt = timedelta(days=1)
        dates = [
            d0 + i * dt for i in range(365)
        ]
        levels = h.get_node(levels_node_path)
        levels = levels[:]
        var_arr = var_arr[:]
        h.close()
        return dates, levels, var_arr

    var_table = h.get_node("/", var_name)

    def grouping_func(myrow):
        return myrow["month"], myrow["day"], myrow["level"]

    date_to_level_to_mean = {}
    date_to_level_to_count = {}

    sel_by_year = "(year >= {0}) & (year <= {1})".format(start_year, end_year)

    #create index on date related columns if it is not created yet
    if not var_table.cols.year.is_indexed:
        var_table.cols.year.createIndex()
        var_table.cols.month.createIndex()
        var_table.cols.day.createIndex()
        var_table.cols.hour.createIndex()

    selection = var_table.where(sel_by_year)

    for gKey, selrows in itertools.groupby(selection, grouping_func):
        the_month, the_day, the_level = gKey
        if the_day == 29 and the_month == 2:
            continue

        d = datetime(2001, the_month, the_day)
        n0 = 0
        x0 = 0
        if d in date_to_level_to_count:
            if the_level in date_to_level_to_count[d]:
                n0 = date_to_level_to_count[d][the_level]
                x0 = date_to_level_to_mean[d][the_level]
        else:
            date_to_level_to_count[d] = {}
            date_to_level_to_mean[d] = {}

        x1 = [
            row["field"] for row in selrows
        ]
        x1 = np.asarray(x1)

        n1 = x1.shape[0]
        x1 = x1.mean(axis=0)
        x1 = (n1 * x1 + n0 * x0) / float(n0 + n1)
        date_to_level_to_mean[d][the_level] = x1
        date_to_level_to_count[d][the_level] = n0 + n1

    sorted_dates = list(sorted(date_to_level_to_mean.keys()))

    sorted_levels = list(sorted(date_to_level_to_mean.items()[0][1].keys()))

    data = np.asarray(
        [[date_to_level_to_mean[d][lev] for lev in sorted_levels] for d in sorted_dates]
    )

    #save cache
    if clim_3d_node not in h:
        h.create_group("/", clim_3d_node[1:])

    h.create_array(clim_3d_node, var_name, data)
    h.create_array(clim_3d_node, "{0}_levels".format(var_name), sorted_levels)
    h.close()
    return sorted_dates, sorted_levels, data


def calculate_daily_mean_fields():
    dates, clim_fields_hcd_rl = Crcm5ModelDataManager.hdf_get_daily_climatological_fields(
        hdf_db_path="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf",
        var_name="STFL", level=None, use_grouping=True, start_year=1979, end_year=1988)

    dates, clim_fields_hcd_rl_intfl = Crcm5ModelDataManager.hdf_get_daily_climatological_fields(
        hdf_db_path="/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup.hdf",
        var_name="STFL", level=None, use_grouping=True, start_year=1979, end_year=1988)

    #Calculate mean timeseries and take a difference
    ts_hcd_rl = []
    for field in clim_fields_hcd_rl:
        field = np.asarray(field)
        ts_hcd_rl.append(field[field >= 0].mean())
    ts_hcd_rl = np.asarray(ts_hcd_rl)

    ts_hcd_rl_intfl = []
    for field in clim_fields_hcd_rl_intfl:
        field = np.asarray(field)
        ts_hcd_rl_intfl.append(field[field >= 0].mean())
    ts_hcd_rl_intfl = np.asarray(ts_hcd_rl_intfl)

    daily_diff_data = (ts_hcd_rl_intfl - ts_hcd_rl) / ts_hcd_rl * 100
    daily_diff_ts = pd.TimeSeries(data=daily_diff_data, index=dates)
    monthly_diff_ts = daily_diff_ts.resample("M", how="mean")

    month_vals = np.asarray([d.month for d in dates])
    month_mean_for_day = np.zeros(len(month_vals))

    fig = plt.figure(figsize=(20, 6))
    ax = plt.gca()
    assert isinstance(ax, Axes)
    ax.set_ylabel("$\left(Q_{\\rm hcd-rl-intfl} - Q_{\\rm hcd-rl}\\right)/Q_{\\rm hcd-rl} \\times 100\%$")
    ax.plot(dates, daily_diff_data)

    ax.plot(ax.get_xlim(), [0, 0], "k-")



    #plot a mean for each month
    for the_month in range(1, 13):
        month_mean_for_day[month_vals == the_month] = monthly_diff_ts[the_month - 1]
        month_dates = list(itertools.ifilter(lambda d: d.month == the_month, dates))
        month_vals = np.ones((len(month_dates),)) * monthly_diff_ts[the_month - 1]
        ax.plot(month_dates, month_vals, "r", lw=1.5)

    ax.grid("on")

    ax.xaxis.set_major_formatter(DateFormatter("%b/%d"))
    ax.xaxis.set_major_locator(DayLocator(bymonthday=1))
    plt.tight_layout()
    plt.savefig("intfl_diff.png")


def get_seasonal_climatology(hdf_path="", start_year=None, end_year=None, var_name="", level=None, months=None):
    #get seasonal climatology, uses daily climatology function which is cached for performance
    #returns the result in m/s
    daily_dates, daily_fields = get_daily_climatology(path_to_hdf_file=hdf_path, var_name=var_name, level=level,
                                                      start_year=start_year, end_year=end_year)

    daily_fields = np.asarray(daily_fields)
    selection_vec = np.array([d.month in months for d in daily_dates], dtype=np.bool)
    return np.mean(daily_fields[selection_vec, :, :], axis=0)


def main():
    manager = Crcm5ModelDataManager(
        samples_folder_path="/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup",
        all_files_in_samples_folder=True)

    hdf_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf"
    mean_field = manager.hdf_get_climatology_for_season(months=[6, 7, 8],
                                                        hdf_db_path=hdf_path,
                                                        var_name="TRAF", level=5,
                                                        start_year=1979,
                                                        end_year=1988)

    plt.contourf(mean_field)
    plt.show()

    pass


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    t0 = time.clock()
    main()
    #calculate_daily_mean_fields()
    print "Elapsed time {0} seconds".format(time.clock() - t0)

    print "Hello world"

