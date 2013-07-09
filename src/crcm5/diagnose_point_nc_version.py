from datetime import datetime
from descartes.patch import PolygonPatch
from matplotlib import gridspec
import matplotlib
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.font_manager import FontProperties
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D, Bbox
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os
import time
import shapely
from shapely.geometry.geo import shape
from shapely.geometry.point import Point
from crcm5.model_data import Crcm5ModelDataManager
from crcm5.model_point import ModelPoint
from cru.temperature import CRUDataManager
from data import cehq_station
from data.cehq_station import Station
from data.cell_manager import CellManager
from swe import SweDataManager

__author__ = 'huziy'

import numpy as np

import fiona
import shapely.prepared as prep

import matplotlib.pyplot as plt


def plot_hydrographs(ax, station, sim_name_to_station_to_model_point,
                     day_stamps=None, sim_names=None):
    """
    Plot climatological hydrographs
    """
    assert isinstance(station, Station)
    assert isinstance(ax, Axes)

    years = station.get_list_of_complete_years()

    #initialize day stamps if it is not passed
    if day_stamps is None:
        day_stamps = Station.get_stamp_days(2001)

    if len(years) < 6:
        return

    handles = []
    labels = []
    dates, obs_data = station.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=day_stamps, years=years)
    obs_ann_mean = np.mean(obs_data)
    label = "Obs: ann.mean = {0:.1f}".format(obs_ann_mean)
    h = ax.plot(dates, obs_data, "k", lw=2, label=label)

    handles.append(h[0])
    labels.append(label)

    mp = None

    for sim_name in sim_names:
        if station in sim_name_to_station_to_model_point[sim_name]:
            continue

        mps = sim_name_to_station_to_model_point[sim_name][station]
        for mp in mps:
            assert isinstance(mp, ModelPoint)
            dates, values = mp.get_daily_climatology_for_complete_years(stamp_dates=day_stamps, varname="STFL")

            label = "{0}: {1:.2f} \n ann.mean = {2:.1f}".format(sim_name,
                                                                mp.mean_upstream_lake_fraction, np.mean(values))
            h = ax.plot(dates, values, label=label, lw=3)

            handles.append(h[0])
            labels.append(label)

    ax.xaxis.set_major_formatter(DateFormatter("%d\n%b"))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1, 13, 3), bymonthday=15))

    if mp is None:
        return
    ax.set_title("{0}: point lake fr.={1:.2f}".format(station.id, mp.lake_fraction))
    return labels, handles


def plot_swe_1d_compare_with_obs(ax, station, sim_name_to_station_to_model_point,
                                 day_stamps=None, sim_names=None):
    """
    Plot climatological swe over upstream points, to the model point corresponding to the station
    sim_name_to_station_to_model_point is a complex dictionary:
    {simulation name: { station: [mp1, mp2, ..., mpN] }}

    sim_name_to_manager:
    {simulation name: Crcm5ModelDataManager object}


    Nad compare with the analysis by Ross Brown
    """
    assert isinstance(station, Station)

    assert isinstance(ax, Axes)

    years = station.get_list_of_complete_years()

    #initialize day stamps if it is not passed
    if day_stamps is None:
        day_stamps = Station.get_stamp_days(2001)

    if len(years) < 6: return {}

    #suppose here that values and times are ordered accordingly in pandas.Timeseries
    obs_data = station.mean_swe_upstream_daily_clim.values
    time = station.mean_swe_upstream_daily_clim.index.to_pydatetime()
    obs_ann_mean = np.mean(obs_data)

    handles = []
    labels = []
    label = "Obs: ann.mean = {0:.1f}".format(obs_ann_mean)
    h = ax.plot(time, obs_data, "k", label=label, lw=2)

    handles.append(h[0])
    labels.append(label)

    ax.set_title("SWE (mm)")
    for sim_name in sim_names:
        if not sim_name_to_station_to_model_point[sim_name].has_key(station):
            continue

        mps = sim_name_to_station_to_model_point[sim_name][station]

        h = None
        for mp in mps:
            assert isinstance(mp, ModelPoint)

            dates, values = mp.get_daily_climatology_for_complete_years(stamp_dates=day_stamps, varname="I5")
            label = "{0}: {1:.2f} \n ann.mean = {2:.1f}".format(sim_name, mp.mean_upstream_lake_fraction
                , np.mean(values))
            h = ax.plot(dates, values, label=label, lw=3)

            handles.append(h[0])
            labels.append(label)

            #ax.xaxis.set_major_formatter(DateFormatter("%d\n%b"))
            #ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,3), bymonthday=15 ))

    return labels, handles


def plot_temp_1d_compare_with_obs(ax, station, sim_name_to_station_to_model_point, sim_names=None):
    assert isinstance(station, Station)

    assert isinstance(ax, Axes)

    years = station.get_list_of_complete_years()

    if len(years) < 6: return None, None

    #suppose here that values and times are ordered accordingly in pandas.Timeseries
    obs_data = station.mean_temp_upstream_monthly_clim.values
    time = station.mean_temp_upstream_monthly_clim.index.to_pydatetime()
    obs_ann_mean = np.mean(obs_data)

    handles = []
    labels = []
    label = "Obs: ann.mean = {0:.1f}".format(obs_ann_mean)
    h = ax.plot(time, obs_data, "k", lw=2, label=label)

    handles.append(h[0])
    labels.append(label)

    ax.set_title("Temperature ")
    for sim_name in sim_names:
        if not sim_name_to_station_to_model_point[sim_name].has_key(station):
            continue

        mps = sim_name_to_station_to_model_point[sim_name][station]

        for mp in mps:
            dates, values = mp.get_monthly_climatology_for_complete_years(varname="TT")
            label = "{0}".format(sim_name)
            h = ax.plot(dates, values, label=label, lw=3)

            handles.append(h[0])
            labels.append(label)

        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1, 13, 3), bymonthday=15))

    return labels, handles


def plot_precip_1d_compare_with_obs(ax, station, sim_name_to_station_to_model_point, sim_names):
    """
    plots precip 1d as subplot
    """
    assert isinstance(station, Station)

    assert isinstance(ax, Axes)

    from calendar import monthrange






    #suppose here that values and times are ordered accordingly in pandas.Timeseries
    obs_data = station.mean_prec_upstream_monthly_clim.values
    time = station.mean_prec_upstream_monthly_clim.index.to_pydatetime()
    obs_ann_mean = np.mean(obs_data)

    handles = []
    labels = []
    label = "Obs: ann.mean = {0:.1f}".format(obs_ann_mean)

    days_in_months = np.array([monthrange(d.year, d.month)[1] for d in time])
    h = ax.plot(time, obs_data / days_in_months, "k", lw=2, label=label) #convert mm/month to mm/day

    handles.append(h[0])
    labels.append(label)

    ax.set_title("Precip mm/day ")
    multiplier = 1000 * 24 * 60 * 60 #to convert from m/s to mm/day
    for sim_name in sim_names:
        if not sim_name_to_station_to_model_point[sim_name].has_key(station):
            continue

        mps = sim_name_to_station_to_model_point[sim_name][station]

        for mp in mps:
            dates, values = mp.get_monthly_climatology_for_complete_years(varname="PR")
            label = "{0}".format(sim_name)
            h = ax.plot(dates, values * multiplier, label=label, lw=3)

            handles.append(h[0])
            labels.append(label)

            #ax.xaxis.set_major_formatter(DateFormatter("%b"))
            #ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,3), bymonthday=15 ))

    return labels, handles


def plot_surf_runoff(ax, station, sim_name_to_station_to_model_point, sim_names=None, day_stamps=None):
    assert isinstance(ax, Axes)


    #initialize day stamps if it is not passed
    if day_stamps is None:
        day_stamps = Station.get_stamp_days(2001)

    ax.plot(day_stamps, [0] * len(day_stamps), "k", lw=0) #so the colors correspond to the same simulation on all panels

    handles = []
    labels = []

    ax.set_title("Surface runoff (${\\rm m^3/s}$)")
    coef = 1.0e-3 #to convert mm to meters
    for sim_name in sim_names:
        if not sim_name_to_station_to_model_point[sim_name].has_key(station):
            continue

        mps = sim_name_to_station_to_model_point[sim_name][station]

        h = None
        for mp in mps:
            the_area = mp.accumulation_area * 1.0e6
            dates, values = mp.get_daily_climatology_for_complete_years(stamp_dates=day_stamps, varname="TRAF")
            label = "{0}".format(sim_name, np.mean(values))
            h = ax.plot(dates, values * the_area * coef, label=label, lw=3)

            handles.append(h[0])
            labels.append(label)

            #ax.xaxis.set_major_formatter(DateFormatter("%d\n%b"))
            #ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,3), bymonthday=15 ))

    return labels, handles


def plot_subsurf_runoff(ax, station, sim_name_to_station_to_model_point, sim_names=None, day_stamps=None):
    assert isinstance(ax, Axes)


    #initialize day stamps if it is not passed
    if day_stamps is None:
        day_stamps = Station.get_stamp_days(2001)

    ax.plot(day_stamps, [0] * len(day_stamps), "k", lw=0) #so the colors correspond to the same simulation on all panels

    handles = []
    labels = []

    ax.set_title("Subsurface runoff (${\\rm m^3/s}$)")
    coef = 1.0e-3 #to convert mm to meters
    for sim_name in sim_names:
        if not sim_name_to_station_to_model_point[sim_name].has_key(station):
            continue

        mps = sim_name_to_station_to_model_point[sim_name][station]

        h = None
        for mp in mps:
            the_area = mp.accumulation_area * 1.0e6
            dates, values = mp.get_daily_climatology_for_complete_years(stamp_dates=day_stamps, varname="TDRA")
            label = "{0}".format(sim_name, np.mean(values))
            h = ax.plot(dates, values * the_area * coef, label=label, lw=3)

            handles.append(h[0])
            labels.append(label)

            #ax.xaxis.set_major_formatter(DateFormatter("%d\n%b"))
            #ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,3), bymonthday=15 ))

    return labels, handles


def plot_total_runoff(ax, station, sim_name_to_station_to_model_point, sim_names=None, day_stamps=None):
    assert isinstance(ax, Axes)


    #initialize day stamps if it is not passed
    if day_stamps is None:
        day_stamps = Station.get_stamp_days(2001)

    ax.plot(day_stamps, [0] * len(day_stamps), "k", lw=0) #so the colors correspond to the same simulation on all panels

    handles = []
    labels = []

    ax.set_title("Total runoff (${\\rm m^3/s}$)")
    coef = 1.0e-3 #to convert mm to meters
    for sim_name in sim_names:
        if not sim_name_to_station_to_model_point[sim_name].has_key(station):
            continue

        mps = sim_name_to_station_to_model_point[sim_name][station]
        h = None
        for mp in mps:
            the_area = mp.accumulation_area * 1.0e6
            dates, tdra = mp.get_daily_climatology_for_complete_years(stamp_dates=day_stamps, varname="TDRA")
            dates, traf = mp.get_daily_climatology_for_complete_years(stamp_dates=day_stamps, varname="TRAF")

            values = traf + tdra
            label = "{0}".format(sim_name, np.mean(values))
            h = ax.plot(dates, values * the_area * coef, label=label, lw=3)

            handles.append(h[0])
            labels.append(label)

            #ax.xaxis.set_major_formatter(DateFormatter("%d\n%b"))
            #ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1,13,3), bymonthday=15 ))

    return labels, handles


def plot_flow_directions_and_basin_boundaries(ax, s, sim_name_to_station_to_model_point,
                                              sim_name_to_manager=None
):
    assert isinstance(ax, Axes)
    assert isinstance(s, Station)
    assert isinstance(sim_name_to_station_to_model_point, dict)

    mp_list = sim_name_to_station_to_model_point.items()[0][1][s]

    #selecting only one (the first model point)
    mp = mp_list[0]

    manager = sim_name_to_manager.items()[0][1]
    assert isinstance(manager, Crcm5ModelDataManager)

    flow_in_mask = manager.get_mask_for_cells_upstream(mp.ix, mp.jy)

    lons2d, lats2d = manager.lons2D, manager.lats2D

    i_upstream, j_upstream = np.where(flow_in_mask == 1)

    nx_total, ny_total = lons2d.shape

    imin, imax = np.min(i_upstream), np.max(i_upstream)
    jmin, jmax = np.min(j_upstream), np.max(j_upstream)

    margin = 8
    imin = max(0, imin - margin)
    jmin = max(0, jmin - margin)
    imax = min(nx_total - 1, imax + margin)
    jmax = min(ny_total - 1, jmax + margin)

    sub_lons2d = lons2d[imin:imax + 1, jmin:jmax + 1]
    sub_lats2d = lats2d[imin:imax + 1, jmin:jmax + 1]
    sub_flow_directions = manager.flow_directions[imin:imax + 1, jmin:jmax + 1]
    sub_flow_in_mask = flow_in_mask[imin:imax + 1, jmin:jmax + 1]

    sub_i_upstream, sub_j_upstream = np.where(sub_flow_in_mask == 1)

    basemap = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(
        lons2d=sub_lons2d, lats2d=sub_lats2d, resolution="h"
    )



    #plot all stations
    #stations = sim_name_to_station_to_model_point.items()[0][1].keys()
    x_list = [the_station.longitude for the_station in (s,)]
    y_list = [the_station.latitude for the_station in (s,)]

    basemap_big = Crcm5ModelDataManager.get_rotpole_basemap_using_lons_lats(
        lons2d=lons2d, lats2d=lats2d
    )
    x_list, y_list = basemap_big(x_list, y_list)

    basemap_big.scatter(x_list, y_list, c="r", s=40, linewidths=0, ax=ax)
    basemap_big.drawcoastlines(ax=ax)
    basemap_big.drawrivers(ax=ax)
    ax.annotate(s.id, xy=basemap(s.longitude, s.latitude), xytext=(3, 3), textcoords='offset points',
                font_properties=FontProperties(weight="bold"), bbox=dict(facecolor='w', alpha=1),
                ha="left", va="bottom", zorder=2)

    x_big, y_big = basemap_big(manager.lons2D, manager.lats2D)


    ####zoom to the area of interest
    #axins = zoomed_inset_axes(ax, 3, loc=2)
    displayCoords = ax.transData.transform((x_big[imin, jmin], y_big[imin, jmin]))
    x_shift_fig, y_shift_fig = ax.figure.transFigure.inverted().transform(displayCoords)
    print "After transData", displayCoords
    print "xshift and yshift", x_shift_fig, y_shift_fig


    #ax.annotate("yuyu", xy = ( 0.733264985153, 0.477182994408), xycoords = "figure fraction" )



    rect = [0.1, y_shift_fig + 0.1, 0.4, 0.4]
    axins = ax.figure.add_axes(rect)


    #assert isinstance(axins, Axes)



    x, y = basemap(sub_lons2d, sub_lats2d)

    x1d_start = x[sub_flow_in_mask == 1]
    y1d_start = y[sub_flow_in_mask == 1]
    fld1d = sub_flow_directions[sub_flow_in_mask == 1]

    from util import direction_and_value

    ishift, jshift = direction_and_value.flowdir_values_to_shift(fld1d)

    sub_i_upstream_next = sub_i_upstream + ishift
    sub_j_upstream_next = sub_j_upstream + jshift

    u = x[sub_i_upstream_next, sub_j_upstream_next] - x1d_start
    v = y[sub_i_upstream_next, sub_j_upstream_next] - y1d_start

    u2d = np.ma.masked_all_like(x)
    v2d = np.ma.masked_all_like(y)

    u2d[sub_i_upstream, sub_j_upstream] = u
    v2d[sub_i_upstream, sub_j_upstream] = v

    basemap.quiver(x, y, u2d, v2d, angles="xy", scale_units="xy", scale=1, ax=axins)

    x_list = [the_station.longitude for the_station in (s,)]
    y_list = [the_station.latitude for the_station in (s,)]
    x_list, y_list = basemap(x_list, y_list)
    basemap.scatter(x_list, y_list, c="r", s=40, linewidths=0)

    basemap.drawcoastlines(ax=axins)
    basemap.drawrivers(ax=axins)


    #read waterbase file, and plot only the polygons which contain at least one upstream point
    shp_path = "/skynet3_exec1/huziy/Netbeans Projects/Java/DDM/data/shape/waterbase/na_bas_ll_r500m/na_bas_ll_r500m.shp"
    c = fiona.open(shp_path)
    hits = c.filter(bbox=(sub_lons2d[0, 0], sub_lats2d[0, 0], sub_lons2d[-1, -1], sub_lats2d[-1, -1]))
    points = [Point(the_x, the_y) for the_x, the_y in zip(x1d_start, y1d_start)]

    selected_polygons = []
    for feature in hits:
        new_coords = []
        old_coords = feature["geometry"]["coordinates"]
        #transform to the map coordinates
        for ring in old_coords:
            x1 = [tup[0] for tup in ring]
            y1 = [tup[1] for tup in ring]
            x2, y2 = basemap(x1, y1)
            new_coords.append(zip(x2, y2))
        feature["geometry"]["coordinates"] = new_coords
        poly = shape(feature["geometry"])
        #print poly, type(poly)
        #print feature.keys()
        #print feature["properties"]
        prep_poly = prep.prep(poly)
        hits = filter(prep_poly.contains, points)
        if len(hits) > 2:
            selected_polygons.append(feature["geometry"])

    for p in selected_polygons:
        axins.add_patch(PolygonPatch(p, fc="none", ec="b", lw=1.5))

    zoom_lines_color = "#6600FF"
    #set color of the frame
    for child in axins.get_children():
        if isinstance(child, Spine):
            child.set_color(zoom_lines_color)
            child.set_linewidth(3)

    # draw a bbox of the region of the inset axes in the parent axes and
    # connecting lines between the bbox and the inset axes area
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec=zoom_lines_color, lw=3)
    #basemap.readshapefile(, "basin")



    pass


def validate_daily_climatology():
    """

    """
    #years are inclusive
    start_year = 1979
    end_year = 1988

    #sim_name_list = ["crcm5-r",  "crcm5-hcd-r", "crcm5-hcd-rl"]
    sim_name_list = ["crcm5-hcd-rl", "crcm5-hcd-rl-intfl"]

    rpn_folder_paths = [
        "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup".format(sim_name_list[0]),
        "/home/huziy/skynet3_rech1/from_guillimin/new_outputs/quebec_0.1_{0}_spinup2/Samples_all_in_one_folder".format(
            sim_name_list[1])
    ]

    nc_db_folder = "/home/huziy/skynet3_rech1/crcm_data_ncdb"


    #select stations
    selected_ids = None
    selected_ids = ["092715", "080101", "074903", "050304", "080104", "081007", "061905",
                    "041903", "040830", "093806", "090613", "081002", "093801", "080718"]

    selected_ids = ["074903", ]

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    selected_ids = None
    stations = cehq_station.read_station_data(selected_ids=selected_ids,
                                              start_date=start_date, end_date=end_date
    )

    stations_hydat = cehq_station.read_hydat_station_data(folder_path="/home/huziy/skynet3_rech1/HYDAT",
                                                          start_date=start_date, end_date=end_date)

    stations.extend(stations_hydat)

    varname = "STFL"
    sim_name_to_manager = {}
    sim_name_to_station_to_model_point = {}

    day_stamps = Station.get_stamp_days(2001)
    sweManager = SweDataManager(var_name="SWE")
    cruTempManager = CRUDataManager(lazy=True)
    cruPreManager = CRUDataManager(var_name="pre", lazy=True,
                                   path="data/cru_data/CRUTS3.1/cru_ts_3_10.1901.2009.pre.dat.nc")

    #common lake fractions when comparing simulations on the same grid
    all_model_points = []

    cell_manager = None

    for sim_name, rpn_folder in zip(sim_name_list, rpn_folder_paths):

        dmManager = Crcm5ModelDataManager(samples_folder_path=rpn_folder, file_name_prefix="dm",
                                          all_files_in_samples_folder=True, need_cell_manager=cell_manager is None)


        #here using the fact that all the simulations are on the same grid
        if cell_manager is None:
            cell_manager = dmManager.cell_manager
        else:
            dmManager.cell_manager = cell_manager



        #determine comon lake fractions, so it is not taken from the trivial case lf = 0, but note
        #this has only sense when all the simulations were performed on the same grid
        sim_name_to_manager[sim_name] = dmManager

        nc_sim_folder = os.path.join(nc_db_folder, sim_name)
        nc_path = os.path.join(nc_sim_folder, "{0}_all.nc4".format(varname))


        #In general there are several model points corresponding to a given station
        st_to_mp = dmManager.get_model_points_for_stations(stations, sim_name=sim_name, nc_path=nc_path,
                                                           nc_sim_folder=nc_sim_folder)

        print "got model points for stations"

        sim_name_to_station_to_model_point[sim_name] = st_to_mp

        #save model points to a list of all points
        for s, mps in st_to_mp.iteritems():
            assert isinstance(s, Station)
            for mp in mps:
                assert isinstance(mp, ModelPoint)
                #calculate upstream swe if needed
                if s.mean_swe_upstream_daily_clim is None:
                    s.mean_swe_upstream_daily_clim = sweManager.get_mean_upstream_timeseries_daily(mp, dmManager,
                                                                                                   stamp_dates=day_stamps)
                    #These are taken from CRU dataset, only monthly data are available
                    s.mean_temp_upstream_monthly_clim = cruTempManager.get_mean_upstream_timeseries_monthly(mp,
                                                                                                            dmManager)
                    s.mean_prec_upstream_monthly_clim = cruPreManager.get_mean_upstream_timeseries_monthly(mp,
                                                                                                           dmManager)

                    print "Calculated observed upstream mean values..."
            all_model_points.extend(mps)

    print "imported input data successfully, plotting ..."


    #for tests
    #test(sim_name_to_station_to_model_point)

    #select only stations which have corresponding model points
    stations = sim_name_to_station_to_model_point[sim_name_list[0]].keys()

    from matplotlib.backends.backend_pdf import PdfPages


    for s in stations:
        years = s.get_list_of_complete_years()
        if len(years) < 6: continue #skip stations with less than 6 continuous years of data

        pp = PdfPages("nc_diagnose_{0}.pdf".format(s.id))

        #plot hydrographs
        fig = plt.figure()
        gs = gridspec.GridSpec(3, 3, left=0.05, hspace=0.3, wspace=0.2)
        ax_stfl = fig.add_subplot(gs[0, 0])
        labels, handles = plot_hydrographs(ax_stfl, s, sim_name_to_station_to_model_point,
                                           day_stamps=day_stamps, sim_names=sim_name_list
        )
        plt.setp(ax_stfl.get_xticklabels(), visible=False) #do not show ticklabels for upper rows

        fig.legend(handles, labels, "lower right")

        #plot swe 1d compare with obs
        ax_swe = fig.add_subplot(gs[1, 0], sharex=ax_stfl)
        plot_swe_1d_compare_with_obs(ax_swe, s, sim_name_to_station_to_model_point,
                                     day_stamps=day_stamps, sim_names=sim_name_list)


        #plot mean temp 1d compare with obs   -- here plot biases directly...??
        ax_temp = fig.add_subplot(gs[0, 2])
        plot_temp_1d_compare_with_obs(ax_temp, s, sim_name_to_station_to_model_point, sim_names=sim_name_list)
        plt.setp(ax_temp.get_xticklabels(), visible=False) #do not show ticklabels for upper rows

        #plot mean precip 1d compare with obs   -- here plot biases directly...??
        ax = fig.add_subplot(gs[1, 2], sharex=ax_temp)
        plot_precip_1d_compare_with_obs(ax, s, sim_name_to_station_to_model_point, sim_names=sim_name_list)


        #plot mean Surface and subsurface runoff
        ax = fig.add_subplot(gs[0, 1], sharex=ax_stfl)
        plot_surf_runoff(ax, s, sim_name_to_station_to_model_point, sim_names=sim_name_list)
        plt.setp(ax.get_xticklabels(), visible=False) #do not show ticklabels for upper rows

        ax = fig.add_subplot(gs[1, 1], sharex=ax_stfl)
        plot_subsurf_runoff(ax, s, sim_name_to_station_to_model_point, sim_names=sim_name_list)
        plt.setp(ax.get_xticklabels(), visible=False) #do not show ticklabels for upper rows

        ax = fig.add_subplot(gs[2, 1], sharex=ax_stfl)
        plot_total_runoff(ax, s, sim_name_to_station_to_model_point, sim_names=sim_name_list)

        pp.savefig()
        #plot flow direction and basin boundaries
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, right=0.99, bottom=0.001)
        ax = fig.add_subplot(gs[0, 1])
        plot_flow_directions_and_basin_boundaries(ax, s, sim_name_to_station_to_model_point,
                                                  sim_name_to_manager=sim_name_to_manager)
        pp.savefig()



        #plot 2d correlation between wind speed and measured streamflow at the station



        pp.close()


def main():
    validate_daily_climatology()
    pass


if __name__ == "__main__":
    t0 = time.time()
    import application_properties
    from util import plot_utils

    plot_utils.apply_plot_params(font_size=18, width_pt=None, height_cm=25, width_cm=39)
    application_properties.set_current_directory()
    main()
    print "Execution time {0} seconds".format(time.time() - t0)
  
