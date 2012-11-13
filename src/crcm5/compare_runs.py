from datetime import datetime, timedelta
import itertools
from matplotlib import gridspec, cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import pandas
from crcm5.model_data import Crcm5ModelDataManager
from data import cehq_station
from data.cehq_station import Station
from data.timeseries import TimeSeries
import my_colormaps
from rpn import level_kinds
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'

import numpy as np
import matplotlib.pyplot as plt





def compare_means_2d(manager_list, start_date = None, end_date = None, months = range(1,13),
                     var_name = "STFL", level = -1 , level_kind = level_kinds.ARBITRARY, out_img = None, impose_min = None,
                     bounds = None):

    """
    Plots a panel of plots for each run
    """

    ncols = 2
    nrows = len(manager_list) // ncols + 1

    #gs = gridspec.GridSpec(nrows, ncols, top=0.985, right=0.985, hspace=0.2, left = 0.01)

    fig = plt.figure()

    vmin = np.Infinity
    vmax = -np.Infinity
    quadMesh = None
    run_id_to_field = {}
    run_id_to_x = {}
    run_id_to_y = {}
    run_id_to_basemap = {}
    run_id_to_manager = {}
    for i, theManager in enumerate( manager_list ):
        assert isinstance(theManager, Crcm5ModelDataManager)
        data_field = theManager.get_mean_field(start_date.year, end_date.year,
            months=months, var_name=var_name, level=level, level_kind=level_kind)

        theId = theManager.run_id

        run_id_to_field[theId] = data_field
        run_id_to_manager[theId] = theManager
        run_id_to_basemap[theId] = theManager.get_omerc_basemap()
        b = run_id_to_basemap[theId]
        x, y = b(theManager.lons2D, theManager.lats2D)
        run_id_to_x[theId] = x
        run_id_to_y[theId] = y

        vmax = max( np.max(data_field), vmax)
        vmin = min( np.min(data_field), vmin )

    if impose_min is not None:
        vmin = impose_min

    print vmin, vmax

    cmap = my_colormaps.get_lighter_jet_cmap(ncolors=15)

    dcol = (vmax - vmin) / float(cmap.N)
    if bounds is None:
        bounds = [vmin + i * dcol for i in range(cmap.N + 1)]

    if vmin * vmax < 0:
        bounds.append(0)
    bounds = list( sorted(bounds) )

    norm = BoundaryNorm( bounds, len(bounds) )

    imgGrid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.5, cbar_size="5%",
        cbar_pad="5%", share_all=True)

    i = 0
    for theId, data_field in run_id_to_field.iteritems():
        ax = imgGrid[i] #fig.add_subplot(gs[i // ncols, i % ncols])
        ax.set_title(theId)
        b = run_id_to_basemap[theId]
        x = run_id_to_x[theId]
        y = run_id_to_y[theId]

        theManager = run_id_to_manager[theId]
        assert isinstance(theManager, Crcm5ModelDataManager)
        if hasattr(theManager, "slope"):
            data_field = np.ma.masked_where( theManager.slope < 0, data_field )

        quadMesh = b.pcolormesh(x, y, data_field, vmin=vmin, vmax = vmax, cmap=cmap, ax = ax, norm=norm)
        b.drawcoastlines(ax = ax)
        i += 1

    assert isinstance(fig, Figure)





    #ax = fig.add_subplot(gs[i // ncols, i % ncols])
    #assert isinstance(ax, Axes)
    #ax.set_aspect(20)
    #ax.set_anchor("W")

    cax = imgGrid.cbar_axes[len(run_id_to_field) - 1]
    cax.colorbar(quadMesh, ticks = bounds, format = "%.3g")

    #fig.colorbar(quadMesh, cax = ax, ticks = bounds, format = "%.2g")
    if out_img is None:
        fig.savefig("_".join(map(str, months))+"_2d_means.png")
    else:
        fig.savefig(out_img)
    pass


def plot_station_positions(manager, station_list):
    """
    :type manager: Crcm5DataManager
    """
    fig = plt.figure()

    ax = fig.add_subplot(1,1,1)
    b = manager.get_omerc_basemap()

    sx = map(lambda s: s.longitude, station_list)
    sy = map(lambda s: s.latitude, station_list)

    sx, sy = b( sx, sy )
    b.scatter(sx, sy, c = "r", s = 40, ax = ax, zorder = 5)
    s_id = map(lambda s: s.id, station_list)

    for x,y, theId in zip(sx, sy, s_id):
        ax.annotate(theId, (x, y), ha = "left", bbox = dict(fc = "w"))
    b.drawcoastlines()
    b.drawrivers()
    fig.savefig("station_pos{0}.png".format("_".join(map(lambda s: s.id, station_list))))
    pass

def compare_hydrographs_at_stations(manager_list, start_date = None, end_date = None, img_path = "hydrographs.png", colors = None):
    selected_ids = None
    stations = cehq_station.read_station_data(selected_ids = selected_ids,
            start_date=start_date, end_date=end_date
    )

    if colors is None:
        colors = len(manager_list) * [None]
    skip_stations = ["080718","095003", "094206", "090613", "092715"]
    #090613 is skipped for the 0.5 deg resolution since the drainaige network is not fully
    #represented by the model

    lines_model = []
    station_to_list_of_model_ts = {}
    run_id_list = [m.run_id for m in manager_list]

    filtered_stations = []
    for s in stations:
        assert isinstance(s, Station)

        if s.id in skip_stations:
            continue

        #skip stations with smaller accumulation areas
        if s.drainage_km2 <= 4 * np.radians(0.5) ** 2 * lat_lon.EARTH_RADIUS_METERS ** 2 * 1.0e-6:
            continue

        if not s.passes_rough_continuity_test(start_date, end_date):
           continue

        filtered_stations.append(s)

    stations = filtered_stations

    #save all run ids
    plot_utils.apply_plot_params(width_pt=None, height_cm =40.0, width_cm=30.0, font_size=10)
    run_id_to_dataframe = {}
    run_id_to_cell_props = {}
    for manager in manager_list:
        assert isinstance(manager, Crcm5ModelDataManager)
        df, station_to_cellprops = manager.get_streamflow_dataframe_for_stations(stations, start_date=start_date,
            end_date=end_date, var_name="STFL", nneighbours=9)
        assert isinstance(df, pandas.DataFrame)
        df = df.dropna(axis = 1)
        run_id_to_cell_props[manager.run_id] = station_to_cellprops


        df = df.groupby(lambda i: datetime(2001,i.month+1, 1)
                                 if i.month == 2 and i.day == 29 else datetime(2001, i.month, i.day) ).mean()

        print df

        #again filter the stations with data time interval overlapping with model time interval
        stations = list( itertools.ifilter(lambda s: s.id in df.columns, stations) )
        run_id_to_dataframe[manager.run_id] = df



    fig = plt.figure()
    #two columns
    ncols = 2
    nrows = len(stations) / ncols
    if nrows * ncols < len(stations):
        nrows += 1
    gs = GridSpec( nrows, ncols, hspace=0.4, wspace=0.4 )
    line_model, line_obs = None, None
    stations.sort(key=lambda x: x.latitude, reverse=True)



    plot_station_positions(manager_list[0], stations)

    i = -1
    ns_list = []
    station_list = []
    flow_acc_area_list = []

    #one_day = timedelta(days = 1)
    one_day_sec = 24 * 60 * 60.0
    for s in stations:
        i += 1
        ax = fig.add_subplot( gs[i // ncols , i % ncols] )

        assert isinstance(s, Station)

        year_dates, sta_clims = s.get_daily_normals()

        #plot error limits
        ax.fill_between(year_dates, sta_clims * 1.256, sta_clims * 0.744, alpha = 0.25, color = "b")
        line_obs = ax.plot(year_dates, sta_clims, color = "b" , label = "Observation", lw = 3, alpha = 0.5)

        ax.annotate("{0:.3g}".format( sum(sta_clims) * one_day_sec ),
            (0.1,0.95), xycoords= "axes fraction", color = "b", alpha = 0.5
        ) #integral flow since those values are daily normals


        for run_id, color, color_index in zip( run_id_list, colors, range(len(colors)) ):
            df = run_id_to_dataframe[run_id]
            the_line = ax.plot(year_dates, df[s.id], color= color, label = run_id, lw = 1)
            ax.annotate("{0:.3g}".format( sum(df[s.id]) * one_day_sec ),
                (0.1,0.9 - color_index * 0.05), xycoords= "axes fraction", color = color
            ) #integral flow since those values are daily normals
            if not i: #save the labels only for the first step
                lines_model.append(the_line)


        #dt = model_ts.time[1] - model_ts.time[0]
        #dt_sec = dt.days * 24 * 60 * 60 + dt.seconds
        #ax.annotate( "{0:g}".format( sum(mod_vals) * dt_sec ) + " ${\\rm m^3}$", xy = (0.7,0.7), xycoords= "axes fraction", color = "b")
        #ax.annotate( "{0:g}".format( sum(s.values) * dt_sec) + " ${\\rm m^3}$", xy = (0.7,0.6), xycoords= "axes fraction", color = "r")
        metadata = run_id_to_cell_props.items()[0][1][s.id]
        da_mod = metadata["acc_area_km2"]
        dist = metadata["distance_to_obs_km"]
        #ax.set_title("{0}: $\delta DA = {1:.1f}$ %, dist = {2:.1f} km".format(s.id,
        #    (da_mod - s.drainage_km2) / s.drainage_km2 * 100.0, dist )  )
        ax.set_title("{0}: $DA = {1:.1f}$ {2}".format(s.id, s.drainage_km2,"${\\rm km ^ 2}$"))
        ax.xaxis.set_major_formatter(DateFormatter("%m"))
        #ax.xaxis.set_major_locator(YearLocator())
        assert isinstance(ax, Axes)
        ax.xaxis.axis_date()
        #ax.xaxis.tick_bottom().set_rotation(60)


    lines = lines_model + [line_obs,]
    labels = run_id_list + ["Observation",]
    fig.legend(lines, labels, ncol = 5)
    fig.savefig(img_path)

def show_lake_effect():
    path_list = [
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v4_old_snc",
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types_crcm_lk_fractions"
    ]

    run_id_list = [
        "w/o lakes, with lake roff.",
        "with lakes, with lake roff."

    ]

    data_managers = []
    for the_path, the_id in zip(path_list, run_id_list):
        theManager = Crcm5ModelDataManager(samples_folder_path=the_path, all_files_in_samples_folder=True)
        theManager.run_id = the_id
        data_managers.append(theManager)

    #compare_hydrographs_at_stations(data_managers, start_date=datetime(1986,1,1), end_date=datetime(1988,12, 31))
    compare_means_2d(data_managers, start_date=datetime(1986,1,1), end_date=datetime(1988,12, 31),
        out_img="lake_effect_2d.png", impose_min=0 )


def show_lake_and_lakeroff_effect():
    path_list = [
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_wo_lakes_and_wo_lakeroff",
        #         "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types_crcm_lk_fractions",
        #         "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_river_ice",
        #         "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_quebec_river_ice_gwrestime0",

                # "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1_lakes_off",
                # "/home/huziy/skynet3_exec1/from_guillimin/quebec_260x260_wo_lakes_and_with_lakeroff",
               #  "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_river_ice_1yrspnp_const_manning",
		 "/home/huziy/skynet3_exec1/from_guillimin/quebec_260x260_wo_lakes_and_with_lakeroff_nogw"

    ]

    run_id_list = [
        "w/o lakes, w/o lake roff.",
     #   "with lakes, with lake roff.",
     #   "with ice",
     #   "with ice, no ground water",

      #  "high res. lakes off",
      #  "high res wo lakes with lake rof",
      #  "low res, const manning, riv ice",

	"high res., nogw, wo lakes, with lakerof"

    ]

    cmap = cm.get_cmap("jet", len(run_id_list))
    #colors = [cmap(i / float(len(run_id_list))) for i in range(len(run_id_list)) ]
    colors = ["r", "k", "g", "y", "m"]

    data_managers = []
    for the_path, the_id in zip(path_list, run_id_list):
        theManager = Crcm5ModelDataManager(samples_folder_path=the_path, all_files_in_samples_folder=True)
        theManager.run_id = the_id
        data_managers.append(theManager)

#    compare_hydrographs_at_stations(data_managers, start_date=datetime(1986,1,1), end_date=datetime(1990,12, 31),
#        img_path="hydrograph_lake_and_lakeroff_effect_test_df.png", colors = colors
#    )
    stfl_bounds = [0, 1, 10, 100, 250, 500, 850, 900, 950, 1000,1200,1500,1800, 2000, 3000, 4000, 10000, 20000]
    compare_means_2d(data_managers, start_date=datetime(1986,1,1), end_date=datetime(1990,12, 31),
        out_img="lake_roff_and_lake_rout_effect_stfl.png", var_name="STFL", level=-1, bounds=stfl_bounds)

#    for month in range(1, 13):
#        compare_means_2d(data_managers, start_date=datetime(1986,1,1), end_date=datetime(1990,12, 31),
#            out_img="lake_roff_and_lake_rout_effect_runoff_month={0}.png".format(month),
#            var_name="TRUN", level=5, bounds=None, months=[month,])


def main():

    path_list = [
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3_old_snc",
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_diff_lk_types_crcm_lk_fractions",
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_without_lakes_v3_sturm_snc",
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1",
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_test_lake_level_260x260_1_lakes_off",
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_with_lakes_bowling_function",
                 "/home/huziy/skynet3_exec1/from_guillimin/quebec_86x86_0.5deg_river_ice"
                 ]
    run_id_list = [
        "w/o lakes",
        "with lakes",
        "w/o lakes sn.cond by Sturm",
        "high res. with lakes",
        "high res. w/o lakes",
        "Bowling lake func.",
        "river ice"
    ]


    data_managers = []
    for the_path, the_id in zip(path_list, run_id_list):
        theManager = Crcm5ModelDataManager(samples_folder_path=the_path, all_files_in_samples_folder=True)
        theManager.run_id = the_id
        data_managers.append(theManager)

    compare_hydrographs_at_stations(data_managers, start_date=datetime(1986,1,1),
        end_date=datetime(1988,12, 31), img_path="hydrographs_all_runs.png")
    
    compare_means_2d(data_managers, start_date=datetime(1986,1,1), end_date=datetime(1988,12, 31), impose_min=0,
        out_img="lake_roff_and_lake_rout_effect_ice.png")
    pass


def plot_mean_2d_fields():
    names = ["STFL",]
    path_list = ["/home/huziy/skynet3_exec1/from_guillimin/quebec_260x260_wo_lakes_and_with_lakeroff",]

    run_id_list = [
        "w/o lakes, with lake roff., high res",

    ]

    data_managers = []
    for the_path, the_id in zip(path_list, run_id_list):
        theManager = Crcm5ModelDataManager(samples_folder_path=the_path, all_files_in_samples_folder=True)
        theManager.run_id = the_id
        data_managers.append(theManager)

    stfl_bounds = [0, 1, 10, 100, 250, 500, 850, 900, 950, 1000,1200,1500,1800, 2000, 3000, 4000, 10000, 20000]
    compare_means_2d(data_managers, start_date=datetime(1986,1,1), end_date=datetime(1990,12, 31),
         out_img="lake_roff_and_lake_rout_effect_stfl_1.png", var_name="STFL", level=-1, bounds=stfl_bounds)

    pass



if __name__ == "__main__":
    import application_properties
    application_properties.set_current_directory()
    #main()
    #show_lake_and_lakeroff_effect()
    plot_utils.apply_plot_params(width_pt=None, font_size=14)
    show_lake_and_lakeroff_effect()
    #show_lake_effect()
    #plot_mean_2d_fields()
    
    print "Hello world"
  
