from datetime import datetime
from netCDF4 import Dataset
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator
from pandas.core.frame import DataFrame
from scipy.spatial.ckdtree import cKDTree
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.cehq_station import Station
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
    kdtree = cKDTree(zip(x0, y0, z0))

    for s in station_list:
        #list of model points which could represent the station

        assert isinstance(s, Station)
        x, y, z = lat_lon.lon_lat_to_cartesian(s.longitude, s.latitude)
        dists, inds = kdtree.query((x, y, z), k=5)

        if npoints == 1:

            deltaDaMin = np.min(np.abs(model_acc_area_1d[inds] - s.drainage_km2))

            #this returns a  list of numpy arrays
            imin = np.where(np.abs(model_acc_area_1d[inds] - s.drainage_km2) == deltaDaMin)[0][0]
            selected_cell_index = inds[imin]
            #check if difference in drainage areas is not too big less than 10 %

            print s.river_name, deltaDaMin / s.drainage_km2
            #if deltaDaMin / s.drainage_km2 > 0.2:
            #    continue

            mp = ModelPoint()
            mp.accumulation_area = model_acc_area_1d[selected_cell_index]
            mp.longitude = lons[selected_cell_index]
            mp.latitude = lats[selected_cell_index]
            mp.cell_index = selected_cell_index
            mp.distance_to_station = dists[imin]

            result[s] = mp
        else:
            raise Exception("npoints = {0}, is not yet implemented ...")
    return result


def main():

    stations = cehq_station.read_grdc_stations(st_id_list=["2903430", "2909150", "2912600", "4208025"])
    statins_to_mp = None

    import matplotlib.pyplot as plt
    labels = ["CanESM", "MPI"]
    paths = ["/skynet3_rech1/huziy/offline_stfl/canesm/discharge_1958_01_01_00_00.nc",
             "/skynet3_rech1/huziy/offline_stfl/mpi/discharge_1958_01_01_00_00.nc"
             ]

    colors = ["r", "b"]


    start_year_current = 1971
    end_year_current = 2000

    start_year_future = 2071
    end_year_future = 2100

    lons2d = None
    lats2d = None


    x_index = None
    y_index = None
    mean_data = None

    sim_to_time = {}


    monthly_dates = [datetime(2001, m, 15) for m in range(1, 13)]
    fmt = DateFormatter("%d\n%b")
    locator = MonthLocator(bymonth=range(2,13,3))

    fig = plt.figure()


    axes = []
    row_indices = []
    col_indices = []

    ncols = 2
    shiftrow = 0 if len(stations) % ncols == 0 else 1
    nrows = len(stations) // ncols + shiftrow
    shared_ax = None
    gs = gridspec.GridSpec(ncols = ncols, nrows= nrows )



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
            ax = fig.add_subplot(gs[row, col], sharey = shared_ax)



        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(fmt)


        assert isinstance(ax, Axes)
        if row < nrows - 1:
            for ticklabel in ax.get_xticklabels():
                ticklabel.set_visible(False)
        if col > 0:
            for ticklabel in ax.get_yticklabels():
                ticklabel.set_visible(False)


        axes.append(ax)





    #plot a panel for each station
    for s, ax, row, col in zip(stations, axes, row_indices, col_indices):

        assert isinstance(s, Station)
        assert isinstance(ax, Axes)
        ax.fill_between(monthly_dates, s.grdc_monthly_clim_min, s.grdc_monthly_clim_max, color="0.6", alpha = 0.5)
        ax.plot(monthly_dates, s.grdc_monthly_clim_mean, "k", lw = 3, label = "Obs")

        ax.set_title(s.river_name)

        for path, sim_label, color in zip(paths, labels, colors):
            ds = Dataset(path)


            if statins_to_mp is None:
                acc_area_2d = ds.variables["accumulation_area"][:]
                lons2d, lats2d = ds.variables["longitude"][:], ds.variables["latitude"][:]
                x_index, y_index = ds.variables["x_index"][:], ds.variables["y_index"][:]
                statins_to_mp = get_dataless_model_points_for_stations(stations, acc_area_2d,
                             lons2d, lats2d, x_index, y_index  )

            #read dates only once for a given simulation
            if not sim_to_time.has_key(sim_label):
                time_str = ds.variables["time"][:]
                times = [ datetime.strptime("".join(t_s), TIME_FORMAT) for t_s in time_str]
                sim_to_time[sim_label] = times



            mp = statins_to_mp[s]
            data = ds.variables["water_discharge_accumulated"][:, mp.cell_index]
            print path
            df = DataFrame(data=data, index = sim_to_time[sim_label], columns=["value"])
            df["year"] = df.index.map(lambda d: d.year)
            df_current = df.ix[df.year.between(start_year_current, end_year_current),:]
            df_future = df.ix[df.year.between(start_year_future, end_year_future),:]



            df_monthly_current = df_current.groupby(by = lambda d: d.month).mean()
            df_monthly_future = df_future.groupby(by = lambda d: d.month).mean()




            monthly_model_current = [ df_monthly_current.ix[month,"value"] for month in range(1,13) ]
            monthly_model_future = [ df_monthly_future.ix[month,"value"] for month in range(1,13) ]





            #print np.mean( monthly_model ), s.river_name, sim_label
            ax.plot(monthly_dates, monthly_model_current, color, lw = 3, label = sim_label + "(C)")
            ax.plot(monthly_dates, monthly_model_future, color+"--", lw=3, label = sim_label + "(F)")


            ds.close()

        if row < nrows - 1:
            ax.set_xticklabels([])

    axes[-1].legend()
    plt.tight_layout()
    plt.savefig("offline_validation.pdf")

if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()

    from util import plot_utils
    plot_utils.apply_plot_params(width_pt=None, width_cm=20, height_cm=20, font_size=18)

    main()
    print "Hello world"
  