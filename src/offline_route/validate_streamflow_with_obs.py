from datetime import datetime, timedelta
from netCDF4 import Dataset
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator, num2date
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
from pandas.core.frame import DataFrame
from rpn.rpn import RPN
from scipy.spatial.ckdtree import cKDTree
from crcm5.compare_runs import plot_station_positions
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.cehq_station import Station
from domains.rotated_lat_lon import RotatedLatLon
from offline_route.plot_seasonal_means import TIME_FORMAT
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np


def get_dataless_model_points_for_stations(station_list, accumulation_area_km2_2d,
                                           model_lons2d, model_lats2d,
                                           i_array, j_array):
    """
    returns a map {station => modelpoint} for comparison modeled streamflows with observed

    this uses exactly the same method for searching model points as one in diagnose_point (nc-version)

    """
    lons = model_lons2d[i_array, j_array]
    lats = model_lats2d[i_array, j_array]
    model_acc_area_1d = accumulation_area_km2_2d[i_array, j_array]
    npoints = 1
    result = {}

    x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lons, lats)
    kdtree = cKDTree(list(zip(x0, y0, z0)))

    for s in station_list:
        # list of model points which could represent the station

        assert isinstance(s, Station)
        x, y, z = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)
        dists, inds = kdtree.query((x, y, z), k=5)

        if npoints == 1:

            deltaDaMin = np.min(np.abs(model_acc_area_1d[inds] - s.drainage_km2))

            # this returns a  list of numpy arrays
            imin = np.where(np.abs(model_acc_area_1d[inds] - s.drainage_km2) == deltaDaMin)[0][0]
            selected_cell_index = inds[imin]
            # check if difference in drainage areas is not too big less than 10 %

            print(s.river_name, deltaDaMin / s.drainage_km2)
            # if deltaDaMin / s.drainage_km2 > 0.2:
            #    continue

            mp = ModelPoint()
            mp.accumulation_area = model_acc_area_1d[selected_cell_index]
            mp.longitude = lons[selected_cell_index]
            mp.latitude = lats[selected_cell_index]
            mp.cell_index = selected_cell_index
            mp.distance_to_station = dists[imin]

            print("Distance to station: ", dists[imin])
            print("Model accumulation area: ", mp.accumulation_area)
            print("Obs accumulation area: ", s.drainage_km2)

            result[s] = mp
        else:
            raise Exception("npoints = {0}, is not yet implemented ...")
    return result


def main():
    # stations = cehq_station.read_grdc_stations(st_id_list=["2903430", "2909150", "2912600", "4208025"])

    selected_ids = ["08MH001", "08NE074", "08NG065", "08NJ013", "08NK002", "08NK016",
                    "08NL004", "08NL007", "08NL024", "08NL038", "08NN002"]
    stations = cehq_station.load_from_hydat_db(natural=True, province="BC", selected_ids=selected_ids)




    stations_to_mp = None

    import matplotlib.pyplot as plt

    # labels = ["CanESM", "MPI"]
    # paths = ["/skynet3_rech1/huziy/offline_stfl/canesm/discharge_1958_01_01_00_00.nc",
    # "/skynet3_rech1/huziy/offline_stfl/mpi/discharge_1958_01_01_00_00.nc"]
    #
    # colors = ["r", "b"]

    # labels = ["ERA", ]
    # colors = ["r", ]
    # paths = ["/skynet3_rech1/huziy/arctic_routing/era40/discharge_1958_01_01_00_00.nc"]


    labels = ["Glacier-only", "All"]
    colors = ["r", "b"]
    paths = [
        "/skynet3_exec2/aganji/glacier_katja/watroute_gemera/discharge_stat_glac_00_99_2000_01_01_00_00.nc",
        "/skynet3_exec2/aganji/glacier_katja/watroute_gemera/discharge_stat_both_00_992000_01_01_00_00.nc"]


    start_year_current = 2000
    end_year_current = 2013

    plot_future = False
    start_year_future = 2071  # ignored when plot future is false
    end_year_future = 2100


    if not plot_future:
        start_year = start_year_current
        end_year = end_year_current
    else:
        start_year = start_year_future
        end_year = end_year_future




    stations_filtered = []
    for s in stations:
        # Also filter out stations with small accumulation areas
        if s.drainage_km2 < 1000:
            continue

        if s.latitude > 49.4:
            continue

        # Filter stations with data out of the required time frame
        year_list = s.get_list_of_complete_years()
        if max(year_list) < start_year or min(year_list) > end_year:
            continue


        stations_filtered.append(s)

    stations = stations_filtered

    min_lon = min(s.longitude for s in stations)
    stations = [s for s in stations if s.longitude == min_lon]


    print("Retained {} stations.".format(len(stations)))

    sim_to_time = {}

    monthly_dates = [datetime(2001, m, 15) for m in range(1, 13)]
    fmt = FuncFormatter(lambda x, pos: num2date(x).strftime("%b")[0])
    locator = MonthLocator()

    fig = plt.figure()

    axes = []
    row_indices = []
    col_indices = []

    ncols = 1
    shiftrow = 0 if len(stations) % ncols == 0 else 1
    nrows = len(stations) // ncols + shiftrow
    shared_ax = None
    gs = gridspec.GridSpec(ncols=ncols, nrows=nrows)

    for i, s in enumerate(stations):
        row = i // ncols
        col = i % ncols

        row_indices.append(row)
        col_indices.append(col)

        if shared_ax is None:
            ax = fig.add_subplot(gs[row, col])
            shared_ax = ax
            assert isinstance(shared_ax, Axes)

        else:
            ax = fig.add_subplot(gs[row, col])

        ax.xaxis.set_major_locator(locator)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

        ax.xaxis.set_major_formatter(fmt)
        sfmt = ScalarFormatter(useMathText=True)
        sfmt.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(sfmt)
        assert isinstance(ax, Axes)

        axes.append(ax)

    # generate daily stamp dates
    d0 = datetime(2001, 1, 1)
    stamp_dates = [d0 + timedelta(days=i) for i in range(365)]



    # plot a panel for each station
    for s, ax, row, col in zip(stations, axes, row_indices, col_indices):

        assert isinstance(s, Station)
        assert isinstance(ax, Axes)
        if s.grdc_monthly_clim_max is not None:
            ax.fill_between(monthly_dates, s.grdc_monthly_clim_min, s.grdc_monthly_clim_max, color="0.6", alpha=0.5)

        avail_years = s.get_list_of_complete_years()
        print("{}: {}".format(s.id, ",".join([str(y) for y in avail_years])))
        years = [y for y in avail_years if start_year <= y <= end_year]
        _, obs_clim_stfl = s.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=stamp_dates, years=years)

        if obs_clim_stfl is None:
            continue

        ax.plot(stamp_dates, obs_clim_stfl, "k", lw=3, label="Obs")

        if s.river_name is not None and s.river_name != "":
            ax.set_title(s.river_name)
        else:
            ax.set_title(s.id)

        for path, sim_label, color in zip(paths, labels, colors):
            ds = Dataset(path)

            if stations_to_mp is None:
                acc_area_2d = ds.variables["accumulation_area"][:]
                lons2d, lats2d = ds.variables["longitude"][:], ds.variables["latitude"][:]
                x_index, y_index = ds.variables["x_index"][:], ds.variables["y_index"][:]
                stations_to_mp = get_dataless_model_points_for_stations(stations, acc_area_2d,
                                                                       lons2d, lats2d, x_index, y_index)

            # read dates only once for a given simulation
            if sim_label not in sim_to_time:
                time_str = ds.variables["time"][:].astype(str)
                times = [datetime.strptime("".join(t_s), TIME_FORMAT) for t_s in time_str]
                sim_to_time[sim_label] = times

            mp = stations_to_mp[s]
            data = ds.variables["water_discharge_accumulated"][:, mp.cell_index]
            print(path)
            df = DataFrame(data=data, index=sim_to_time[sim_label], columns=["value"])
            df["year"] = df.index.map(lambda d: d.year)
            df = df.ix[df.year.isin(years), :]
            df = df.select(lambda d: not (d.month == 2 and d.day == 29))
            df = df.groupby(lambda d: datetime(stamp_dates[0].year, d.month, d.day)).mean()

            daily_model_data = [df.ix[d, "value"] for d in stamp_dates]

            # print np.mean( monthly_model ), s.river_name, sim_label
            ax.plot(stamp_dates, daily_model_data, color, lw=3, label=sim_label + "(C)")

            if plot_future:
                ax.plot(stamp_dates, daily_model_data, color + "--", lw=3, label=sim_label + "(F2)")

            ds.close()

        if row < nrows - 1:
            ax.set_xticklabels([])

    axes[0].legend(fontsize=17, loc=2)
    plt.tight_layout()
    plt.savefig("offline_validation.png", dpi=400)
    plt.close(fig)


    r = RPN("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/Depth_to_bedrock_WestNA_0.25")
    r.get_first_record_for_name("8L")
    proj_params = r.get_proj_parameters_for_the_last_read_rec()
    lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    bsmp = RotatedLatLon(**proj_params).get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)
    plot_utils.apply_plot_params(width_pt=None, width_cm=19, height_cm=19, font_size=12)
    plot_station_positions(manager=None, station_list=stations, bsmp=bsmp)




if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()

    from util import plot_utils

    plot_utils.apply_plot_params(width_pt=None, width_cm=19, height_cm=40, font_size=22)

    main()
    print("Hello world")