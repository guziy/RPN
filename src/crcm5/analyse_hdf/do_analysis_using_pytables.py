from datetime import datetime

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


def get_basemap_from_hdf(file_path = ""):
    """
    :param file_path:
    :return: lons(2d), lats(2), basemap - corresponding to the data in the file
    """
    h = tb.open_file(file_path)

    # Extract 2d longitudes and latitudes
    lons = h.getNode("/", "longitude")[:]
    lats = h.getNode("/", "latitude")[:]

    rotPoleTable = h.getNode("/", "rotpole")

    assert isinstance(rotPoleTable, tb.Table)

    params = {}
    for row in rotPoleTable:
        params[row["name"]] = row["value"]
    rotPoleTable.close()


    basemap = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(
        lons2d=lons, lats2d=lats,
        lon_1=params["lon1"], lon_2=params["lon2"],
        lat_1=params["lat1"], lat_2=params["lat2"]
    )
    h.close()
    return lons, lats, basemap


def get_array_from_file(path = "", var_name = ""):
    h = tb.open_file(path)

    if not var_name in h.root:
        print "Warning: no {0} in {1}".format(var_name, path)
        return None

    data = h.get_node("/", var_name)[:]
    h.close()
    return data


def get_mean_2d_fields_for_months(path = "", var_name = "", level = None, months = None,
                                  start_year = None,
                                  end_year = None):
    """
    Return means over the specified months for each year
    :param path: path to the hdf file with data
    :param var_name:
    :param level:
    """
    return Crcm5ModelDataManager.hdf_get_seasonal_means(path_to_hdf=path, months=months, var_name=var_name, level=level,
                                                        start_year=start_year, end_year=end_year)





def get_daily_means_for_a_point(path = "", var_name = "STFL", level = None,
                                years_of_interest = None,
                                i_index = None, j_index = None):
    """
    :param years_of_interest: is a list of years used for calculating daily climatologies
    """
    h = tb.open_file(path)

    varTable = h.get_node("/", var_name)

    def grouping_func(row):
        return row["month"], row["day"], row["level"]

    date_to_mean = {}
    date_to_count = {}

    if level is not None:
        selByLevel = "(level == {0})".format(level)
    else:
        selByLevel = ""

    selByYear = "|".join(["(year == {0})".format(y) for y in years_of_interest])

    if level is not  None:
        selForYearsAndLevel = varTable.where(
            "({0}) & ({1}) & ~((month == 2) & (day == 29))".format(selByLevel, selByYear) )
    else:
        print "({0}) & ((month != 2) | (day != 29))".format(selByYear)
        selForYearsAndLevel = varTable.where(
            "({0}) & ((month != 2) | (day != 29))".format(selByYear))


    for gKey, selrows in itertools.groupby(selForYearsAndLevel, grouping_func):
        the_month, the_day, the_level = gKey
        d = datetime(2001, the_month, the_day)
        n0 = 0
        x0 = 0
        if d in date_to_count:
            n0 = date_to_count[d]
            x0 = date_to_mean[d]

        x1 = np.asarray([row["field"][i_index, j_index] for row in selrows])
        n1 = x1.shape[0]
        x1 = x1.mean(axis = 0)
        x1 = (n1 * x1 + n0 * x0) / float(n0 + n1)
        date_to_mean[d] = x1
        date_to_count[d] = n0 + n1

    h.close()
    sorted_dates = list( sorted(date_to_mean.keys()) )
    return sorted_dates, [date_to_mean[d] for d in sorted_dates]



def get_daily_climatology(path_to_hdf_file = "", var_name = "STFL", level = None, start_year = None, end_year = None):
    return Crcm5ModelDataManager.hdf_get_daily_climatological_fields(
        hdf_db_path=path_to_hdf_file, start_year=start_year, end_year=end_year, var_name=var_name,
        level=level, use_grouping=True)



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
    daily_diff_ts = pd.TimeSeries(data = daily_diff_data, index = dates)
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
        ax.plot(month_dates, month_vals, "r", lw = 1.5)



    ax.grid("on")

    ax.xaxis.set_major_formatter(DateFormatter("%b/%d"))
    ax.xaxis.set_major_locator(DayLocator(bymonthday=1))
    plt.tight_layout()
    plt.savefig("intfl_diff.png")

    pass

def get_seasonal_climatology(hdf_path = "", start_year = None, end_year = None, var_name = "", level = None, months = None):
    #get seasonal climatology, uses daily climatology function which is cached for performance
    #returns the result in m/s
    daily_dates, daily_fields = get_daily_climatology(path_to_hdf_file=hdf_path, var_name=var_name, level = level,
        start_year=start_year, end_year=end_year)

    daily_fields = np.asarray(daily_fields)
    selection_vec = np.array([d.month in months for d in daily_dates], dtype=np.bool)
    return np.mean(daily_fields[selection_vec, :, :] , axis = 0)


def main():
    manager = Crcm5ModelDataManager(
        samples_folder_path="/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_crcm5-r_spinup",
        all_files_in_samples_folder=True)

    hdf_path = "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf"
    mean_field = manager.hdf_get_climatology_for_season(months=[6, 7, 8],
                                                        hdf_db_path= hdf_path,
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

