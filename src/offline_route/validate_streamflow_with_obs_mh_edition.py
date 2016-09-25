from datetime import datetime, timedelta
from netCDF4 import Dataset
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator, num2date
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
from pandas.core.frame import DataFrame
from pathlib import Path
from rpn.rpn import RPN
from scipy.spatial.ckdtree import cKDTree
from crcm5.compare_runs import plot_station_positions
from crcm5.model_point import ModelPoint
import data.cehq_station as cehq_station
from data.cehq_station import Station
from data.cell_manager import CellManager
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


        if npoints == 1:

            if s.drainage_km2 is not None:
                dists, inds = kdtree.query((x, y, z), k=10)
                deltaDaMin = np.min(np.abs(model_acc_area_1d[inds] - s.drainage_km2))

                # this returns a  list of numpy arrays
                imin = np.where(np.abs(model_acc_area_1d[inds] - s.drainage_km2) == deltaDaMin)[0][0]
                selected_cell_index = inds[imin]
                # check if difference in drainage areas is not too big less than 10 %

                print(s.river_name, deltaDaMin / s.drainage_km2)
                # if deltaDaMin / s.drainage_km2 > 0.2:
                #    continue
            else:
                dists, inds = kdtree.query((x, y, z), k=1)
                selected_cell_index = inds
                imin = 0
                dists = [dists]



            mp = ModelPoint(ix=i_array[selected_cell_index], jy=j_array[selected_cell_index])
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

    selected_station_ids = [
        "05LM006",
        "05BN012",
        "05AK001",
        "05QB003",
        "06EA002"
    ]

    stations = cehq_station.load_from_hydat_db(natural=None, province=None, selected_ids=selected_station_ids, skip_data_checks=True)

    stations_mh = cehq_station.get_manitoba_hydro_stations()

    # copy metadata from the corresponding hydat stations
    for s in stations:
        assert isinstance(s, Station)
        for s_mh in stations_mh:
            assert isinstance(s_mh, Station)


            if s == s_mh:
                s_mh.copy_metadata(s)
                break



    stations = [s for s in stations_mh if s.id in selected_station_ids and s.longitude is not None]

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


    labels = ["Model", ]
    colors = ["r", ]
    paths = [
        "/RESCUE/skynet3_rech1/huziy/water_route_mh_bc_011deg_wc/discharge_1980_01_01_12_00.nc"
    ]

    infocell_path = "/RESCUE/skynet3_rech1/huziy/water_route_mh_bc_011deg_wc/infocell.nc"

    start_year = 1980
    end_year = 2014




    stations_filtered = []
    for s in stations:
        # Also filter out stations with small accumulation areas
        # if s.drainage_km2 is not None and s.drainage_km2 < 100:
        #     continue

        # Filter stations with data out of the required time frame
        year_list = s.get_list_of_complete_years()

        print("Complete years for {}: {}".format(s.id, year_list))

        stations_filtered.append(s)

    stations = stations_filtered


    print("Retained {} stations.".format(len(stations)))

    sim_to_time = {}

    monthly_dates = [datetime(2001, m, 15) for m in range(1, 13)]
    fmt = FuncFormatter(lambda x, pos: num2date(x).strftime("%b")[0])
    locator = MonthLocator(bymonthday=15)

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
        obs_clim_stfl = s.get_monthly_climatology(years_list=years)

        if obs_clim_stfl is None:
            continue

        print(obs_clim_stfl.head())

        obs_clim_stfl.plot(color="k", lw=3, label="Obs", ax=ax)

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
            df = df.groupby(lambda d: datetime(2001, d.month, 15)).mean()


            # print np.mean( monthly_model ), s.river_name, sim_label
            df.plot(color=color, lw=3, label=sim_label, ax=ax, y="value")


            ds.close()

        if row < nrows - 1:
            ax.set_xticklabels([])

    axes[0].legend(fontsize=17, loc=2)
    plt.tight_layout()
    plt.savefig("mh/offline_validation_mh.png", dpi=400)
    plt.close(fig)






    with Dataset(infocell_path) as ds:

        fldir = ds.variables["flow_direction_value"][:]
        faa = ds.variables["accumulation_area"][:]

        lon, lat = [ds.variables[k][:] for k in ["lon", "lat"]]

        # plot station positions and upstream areas
        cell_manager = CellManager(fldir, nx=fldir.shape[0], ny=fldir.shape[1],
                                   lons2d=lon, lats2d=lat, accumulation_area_km2=faa)



    fig = plt.figure()
    from crcm5.mh_domains import default_domains
    gc = default_domains.bc_mh_011

    # get the basemap object
    bmp, data_mask = gc.get_basemap_using_shape_with_polygons_of_interest(
        lon, lat, shp_path=default_domains.MH_BASINS_PATH, mask_margin=5)

    xx, yy = bmp(lon, lat)
    ax = plt.gca()
    colors = ["g", "r", "m", "c", "y", "violet"]
    i = 0
    for s, mp in stations_to_mp.items():
        assert isinstance(mp, ModelPoint)
        upstream_mask = cell_manager.get_mask_of_upstream_cells_connected_with_by_indices(mp.ix, mp.jy)

        current_points = upstream_mask > 0.5

        bmp.drawcoastlines()
        bmp.drawrivers()

        bmp.scatter(xx[current_points], yy[current_points], c=colors[i % len(colors)])
        i += 1


        va = "top"
        if s.id in ["05AK001", "05LM006"]:
            va = "bottom"

        ha = "left"
        if s.id in ["05QB003"]:
            ha = "right"

        bmp.scatter(xx[mp.ix, mp.jy], yy[mp.ix, mp.jy], c="b")
        ax.annotate(s.id, xy=(xx[mp.ix, mp.jy], yy[mp.ix, mp.jy]), horizontalalignment=ha,
                    verticalalignment=va, bbox=dict(boxstyle='round', fc='gray', alpha=0.5))

    fig.savefig("mh/offline_stations_{}.png".format("positions"))
    plt.close(fig)


    # r = RPN("/RESCUE/skynet3_rech1/huziy/CNRCWP/C3/Depth_to_bedrock_WestNA_0.25")
    # r.get_first_record_for_name("8L")
    # proj_params = r.get_proj_parameters_for_the_last_read_rec()
    # lons, lats = r.get_longitudes_and_latitudes_for_the_last_read_rec()
    # bsmp = RotatedLatLon(**proj_params).get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats)
    # plot_utils.apply_plot_params(width_pt=None, width_cm=19, height_cm=19, font_size=12)
    # plot_station_positions(manager=None, station_list=stations, bsmp=bsmp)




if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()

    from util import plot_utils

    plot_utils.apply_plot_params(width_pt=None, width_cm=19, height_cm=40, font_size=10)

    main()
    print("Hello world")