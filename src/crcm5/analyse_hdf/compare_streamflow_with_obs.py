from collections import OrderedDict
from datetime import datetime
import itertools
from matplotlib import cm
import brewer2mpl
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.figure import Figure
import os
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap
import pandas
from pandas.tseries.converter import _daily_finder
from crcm5 import infovar, model_point
from crcm5.analyse_hdf.plot_station_positions import plot_positions_of_station_list
from crcm5.model_point import ModelPoint
from data import cehq_station
from data.anusplin import AnuSplinManager
from data.cehq_station import Station
from data.cell_manager import CellManager
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import numpy as np
from swe import SweDataManager

__author__ = 'huziy'

import common_plot_params as cpp

import do_analysis_using_pytables as analysis
#TODO: this is to compare streamflow simulation results with station data
#   input: station ids or list of station objects and
#          the list of simulations, to compare with

#Notes: All the model simulations are assumed to be on the same grid

images_folder = "/home/huziy/skynet3_rech1/Netbeans Projects/Python/RPN/images_for_lake-river_paper"


def _plot_station_position(ax, the_station, basemap, cell_manager, the_model_point):
    assert the_station is None or isinstance(the_station, Station)
    assert isinstance(the_model_point, ModelPoint)
    assert isinstance(cell_manager, CellManager)

    x, y = basemap(cell_manager.lons2d, cell_manager.lats2d)


    #plot the arrows for upstream cells
    ups_mask = cell_manager.get_mask_of_cells_connected_with_by_indices(the_model_point.ix, the_model_point.jy)

    x1d_start = x[ups_mask == 1]
    y1d_start = y[ups_mask == 1]
    fld1d = cell_manager.flow_directions[ups_mask == 1]
    i_upstream, j_upstream = np.where(ups_mask == 1)

    imin, jmin = i_upstream.min() - 60, j_upstream.min() - 60
    imax, jmax = i_upstream.max() + 40, j_upstream.max() + 40

    nx, ny = x.shape
    imax, jmax = min(imax, nx - 1), min(jmax, ny - 1)
    imin, jmin = max(imin, 0), max(jmin, 0)

    basemap_initial_corners = [
        basemap.llcrnrx, basemap.llcrnry,
        basemap.urcrnrx, basemap.urcrnry
    ]

    basemap.llcrnrx = x[imin, jmin]
    basemap.llcrnry = y[imin, jmin]
    basemap.urcrnrx = x[imax, jmax]
    basemap.urcrnry = y[imax, jmax]

    if the_station is not None:
        x_station, y_station = basemap(the_station.longitude, the_station.latitude)
    else:
        x_station, y_station = basemap(the_model_point.longitude, the_model_point.latitude)

    basemap.scatter(x_station, y_station, c="b", s=25, ax=ax, linewidths=0, zorder=2)

    from util import direction_and_value

    ishift, jshift = direction_and_value.flowdir_values_to_shift(fld1d)

    sub_i_upstream_next = i_upstream + ishift
    sub_j_upstream_next = j_upstream + jshift

    u = x[sub_i_upstream_next, sub_j_upstream_next] - x1d_start
    v = y[sub_i_upstream_next, sub_j_upstream_next] - y1d_start

    u2d = np.ma.masked_all_like(x)
    v2d = np.ma.masked_all_like(y)

    u2d[i_upstream, j_upstream] = u
    v2d[i_upstream, j_upstream] = v

    #basemap.quiver(x, y, u2d, v2d, angles="xy", scale_units="xy", scale=1, ax=ax)
    basemap.pcolormesh(x, y, np.ma.masked_where(ups_mask < 0.5, ups_mask) * 0.5, cmap=cm.get_cmap(name="gray"),
                       ax=ax, vmax=1, vmin=0)

    basemap.drawcoastlines(linewidth=cpp.COASTLINE_WIDTH)

    #put back the initial corners of the basemap
    basemap.llcrnrx, basemap.llcrnry, basemap.urcrnrx, basemap.urcrnry = basemap_initial_corners

    return ups_mask


#noinspection PyNoneFunctionAssignment
def _validate_temperature_with_anusplin(ax, the_model_point, model_data_dict=None,
                                        obs_tmin_clim_fields=None,
                                        obs_tmax_clim_fields=None, daily_dates=None,
                                        cell_area_km2=None,
                                        upstream_mask=None,
                                        simlabel_list=None):
    # model_data_dict - is a dictionary {sim. label: daily climatological fields}
    # obs_tmin_clim_fields - daily climatological fields
    # The order of plotting follows simlabel_list and the observations are plotted the last

    assert isinstance(ax, Axes)
    assert isinstance(the_model_point, ModelPoint)

    good_points = (upstream_mask == 1) & (~obs_tmin_clim_fields[0].mask)
    basin_area_km2 = np.sum(cell_area_km2[good_points])
    area_matrix = cell_area_km2 * good_points.astype(int)

    ax.set_ylabel(r"2m temperature: ${\rm ^\circ C}$")

    resample_period = "5D"  # 5 day running mean

    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2
        df = _apply_running_mean(daily_dates, data, averaging_period=resample_period)
        ax.plot(df.index, df["values"], label=label, lw=1)

    i_select, j_select = np.where(good_points)

    if not np.any(np.isnan(obs_tmax_clim_fields[0][upstream_mask == 1])):
        basin_tmax = np.sum(obs_tmax_clim_fields[:, i_select, j_select] * area_matrix[np.newaxis, i_select, j_select],
                            axis=1) / basin_area_km2
        basin_tmin = np.sum(obs_tmin_clim_fields[:, i_select, j_select] * area_matrix[np.newaxis, i_select, j_select],
                            axis=1) / basin_area_km2

        #plot max temperature
        df_tmax = _apply_running_mean(daily_dates, basin_tmax)
        print df_tmax.tail(20)
        p = ax.plot(df_tmax.index, df_tmax["values"], label="tmax-obs", lw=1)

        #plot min temperature
        df_tmin = _apply_running_mean(daily_dates, basin_tmin)
        ax.plot(df_tmin.index, df_tmin["values"], label="tmin-obs", color=p[0].get_color(), lw=1)
        ax.fill_between(df_tmin.index, df_tmax["values"], df_tmin["values"], alpha=0.3, color=p[0].get_color())

        print "--" * 20
        print "tmin ranges: from {0} to {1}".format(np.min(basin_tmin), np.max(basin_tmin))
        print "tmin ranges: from {0} to {1}".format(np.min(df_tmin["values"]), np.max(df_tmin["values"]))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_label_position("right")
    ax.grid()


#noinspection PyNoneFunctionAssignment
def _validate_precip_with_anusplin(ax, the_model_point, model_data_dict=None,
                                   obs_precip_clim_fields=None,
                                   daily_dates=None,
                                   cell_area_km2=None,
                                   upstream_mask=None,
                                   simlabel_list=None):
    # model_data_dict - is a dictionary {sim. label: daily climatological fields}
    # obs_tmin_clim_fields - daily climatological fields
    # The order of plotting follows simlabel_list and the observations are plotted the last

    assert isinstance(ax, Axes)
    assert isinstance(the_model_point, ModelPoint)

    good_points = (upstream_mask == 1) & (~obs_precip_clim_fields[0].mask)

    basin_area_km2 = np.tensordot(cell_area_km2, upstream_mask)
    area_matrix = cell_area_km2 * good_points.astype(int)

    ax.set_ylabel(r"Total precip.: mm/day")

    resample_period = "5D"
    i_select, j_select = np.where(good_points)
    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2

        #convert m/s to mm/day
        data = data * 1000.0 * 24 * 60 * 60

        df = _apply_running_mean(daily_dates, data, averaging_period=resample_period)

        ax.plot(df.index, df["values"], label=label, lw=1)


    #obs_precip_clim_fields[np.isnan(obs_precip_clim_fields)] = -9999
    #Do not draw region averaged observed precip for the regions, where there is missing data
    if not np.any(np.isnan(obs_precip_clim_fields[0][upstream_mask == 1])):
        basin_precip = np.sum(
            obs_precip_clim_fields[:, i_select, j_select] * area_matrix[np.newaxis, i_select, j_select],
            axis=1) / basin_area_km2

        #running mean
        df = _apply_running_mean(daily_dates, basin_precip, averaging_period=resample_period)
        ax.plot(df.index, df["values"], label="precip-obs", lw=2)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_label_position("right")
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid()


#noinspection PyNoneFunctionAssignment
def _validate_swe_with_ross_brown(ax, the_model_point, model_data_dict=None,
                                  obs_swe_clim_fields=None,
                                  daily_dates=None,
                                  cell_area_km2=None,
                                  upstream_mask=None,
                                  simlabel_list=None):
    # model_data_dict - is a dictionary {sim. label: daily climatological fields}
    # obs_tmin_clim_fields - daily climatological fields
    # The order of plotting follows simlabel_list and the observations are plotted the last

    assert isinstance(ax, Axes)
    assert isinstance(the_model_point, ModelPoint)

    basin_area_km2 = np.tensordot(cell_area_km2, upstream_mask)
    area_matrix = upstream_mask * cell_area_km2

    basin_swe = np.tensordot(obs_swe_clim_fields, area_matrix) / basin_area_km2

    ax.set_ylabel(r"swe: mm")

    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2
        ax.plot(daily_dates, data, label=label, lw=2)
        dv = (np.max(data) - np.max(basin_swe)) * basin_area_km2 * 1.0e-3 * 1.0e6
        print dv, np.max(data), np.max(basin_swe), basin_area_km2
        print "{}, [{}]: dSWE_max * A={}".format(label, the_model_point.point_id, dv)

    p = ax.plot(daily_dates, basin_swe, label="swe-obs", lw=2)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    #plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid()


def _apply_running_mean(index, values, averaging_period="5D"):
    """

    :param index:
    :param values:
    :param averaging_period:
    :return: df - resampled pandas.DataFrame
    """
    df = pandas.DataFrame(index=index, data=values, columns=["values"])
    df = df.resample(averaging_period, how=np.mean)
    return df[:-1]  # do not consider the last one because it might go beyond the limits


def calclulate_spring_peak_err(dates, qobs, qmod, st_id=None, da_obs=None, da_mod=None):
    dt = dates[1] - dates[0]
    dt = dt.total_seconds()



    #select spring
    s = sum((qmodi - qobsi) * dt for qobsi, qmodi, t in zip(qobs, qmod, dates)
            if t.month in (4, 5, 6) and qmodi - qobsi*da_mod / da_mod >= 0)
    print "{}: int(dqdt) = {}".format(st_id, s)



#noinspection PyNoneFunctionAssignment
def _plot_upstream_surface_runoff(ax, the_model_point, model_data_dict=None,
                                  daily_dates=None,
                                  cell_area_km2=None,
                                  upstream_mask=None,
                                  simlabel_list=None):
    #plot arae averaged upstream surface runoff

    assert isinstance(ax, Axes)
    assert isinstance(the_model_point, ModelPoint)

    basin_area_km2 = np.tensordot(cell_area_km2, upstream_mask)
    area_matrix = upstream_mask * cell_area_km2

    ax.set_ylabel("Surf. r-off: mm/day")

    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2
        #convert mm/s to mm/day
        data = data * 24 * 60 * 60
        data[data < 0] = 0  # set negative surface runoff to 0
        df = _apply_running_mean(daily_dates, data)
        ax.plot(df.index, df["values"], label=label, lw=1)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_label_position("right")
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid()


#noinspection PyNoneFunctionAssignment
def _plot_upstream_subsurface_runoff(ax, the_model_point, model_data_dict=None,
                                     daily_dates=None,
                                     cell_area_km2=None,
                                     upstream_mask=None,
                                     simlabel_list=None):
    #plot arae averaged upstream surface runoff

    assert isinstance(ax, Axes)
    assert isinstance(the_model_point, ModelPoint)

    basin_area_km2 = np.tensordot(cell_area_km2, upstream_mask)
    area_matrix = upstream_mask * cell_area_km2

    ax.set_ylabel("Drainage: mm/day")
    ax.yaxis.set_label_position("right")

    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2
        #convert mm/s to mm/day
        data = data * 24 * 60 * 60

        df = _apply_running_mean(daily_dates, data)
        ax.plot(df.index, df["values"], label=label, lw=1)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.grid()


def write_annual_flows_to_txt(sim_label_list, sim_to_values_model, values_obs, file_annual_discharge, **kwargs):

    """
    line_format has the following form

    line_format = "{0:10s}\t{1:10.1f}\t{1:10.1f}\t" + "\t".join(["{" + str(i + 3) + ":10.1f}"
                                                                 for i in range(len(sim_name_to_file_name))])
    i.e. Station id, DAobs, DAmod, ValForSim1, ValForSim2,..., ValForSimn

    :param sim_to_values_model:
    :param values_obs:
    :param file_annual_discharge:
    """


    st_id = kwargs["station_id"]
    da_obs = kwargs["da_obs"]
    da_mod = kwargs["da_mod"]


    qmod_km3 = [sum(sim_to_values_model[label]) * 24 * 60 * 60 * 1.0e-9 for label in sim_label_list]
    qobs_km3 = sum(values_obs) * 24 * 60 * 60 * 1.0e-9

    q_all = qmod_km3 + [qobs_km3, ]
    line_data = [st_id, da_obs, da_mod] + q_all

    print "da_obs={}; da_mod={}".format(da_obs, da_mod)

    line_format = "{0:10s}\t{1:10.1f}\t{2:10.1f}\t" + "\t".join(["{" + str(i + 3) + ":10.1f}"
                                                                 for i in range(len(sim_label_list) + 1)]) + "\n"
    file_annual_discharge.write(line_format.format(*line_data))



def draw_model_comparison(model_points=None, stations=None, sim_name_to_file_name=None, hdf_folder=None,
                          start_year=None, end_year=None, cell_manager=None, stfl_name="STFA",
                          drainage_area_reldiff_min=0.1):
    """

    :param model_points: list of model point objects
    :param stations: list of stations corresponding to the list of model points
    :param cell_manager: is a CellManager instance which can be provided for better performance if necessary
    len(model_points) == len(stations) if stations is not None.
    if stations is None - then no measured streamflow will be plotted
    """
    assert model_points is None or stations is None or len(stations) == len(model_points)
    label_list = list(sim_name_to_file_name.keys())  # Needed to keep the order the same for all subplots
    path0 = os.path.join(hdf_folder, sim_name_to_file_name.items()[1][1])
    flow_directions = analysis.get_array_from_file(path=path0, var_name="flow_direction")
    lake_fraction = analysis.get_array_from_file(path=path0, var_name="lake_fraction")

    accumulation_area_km2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_ACCUMULATION_AREA_NAME)
    area_m2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_CELL_AREA_NAME_M2)

    #Try to read cell areas im meters if it is not Ok then try in km2
    if area_m2 is not None:
        cell_area_km2 = area_m2 * 1.0e-6
    else:
        cell_area_km2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_CELL_AREA_NAME_KM2)


    print "cell area ranges from {} to {}".format(cell_area_km2.min(), cell_area_km2.max())

    # print "plotting from {0}".format(path0)
    # plt.pcolormesh(lake_fraction.transpose())
    # plt.colorbar()
    # plt.show()
    # exit()

    file_scores = open("scores_{0}_{1}-{2}.txt".format("_".join(label_list), start_year, end_year), "w")
    file_correlations = open("corr_{0}_{1}-{2}.txt".format("_".join(label_list), start_year, end_year), "w")
    file_annual_discharge = open("flow_{0}_{1}-{2}.txt".format("_".join(label_list), start_year, end_year), "w")

    text_files = [file_scores, file_correlations, file_annual_discharge]
    #write the following columns to the scores file
    header_format = "{0:10s}\t{1:10s}\t{2:10s}\t" + "\t".join(["{" + str(i + 3) + ":10s}"
                                                               for i in range(len(sim_name_to_file_name))])
    line_format = "{0:10s}\t{1:10.1f}\t{2:10.1f}\t" + "\t".join(["{" + str(i + 3) + ":10.1f}"
                                                                 for i in range(len(sim_name_to_file_name))])

    header_ns = ("ID", "DAo", "DAm",) + tuple(["NS({0})".format(key) for key in sim_name_to_file_name])
    file_scores.write(header_format.format(*header_ns) + "\n")

    header_qyear = ("ID", "DAo", "DAm",) + tuple(["Qyear({0})".format(key) for key in label_list]) +\
                   ("Qyear(obs)",)
    header_format_qyear = header_format + "\t{" + str(len(label_list) + 3) + ":10s}"
    file_annual_discharge.write(header_format_qyear.format(*header_qyear) + "\n")


    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=path0)

    #Create a cell manager if it is not provided
    if cell_manager is None:
        cell_manager = CellManager(flow_directions, accumulation_area_km2=accumulation_area_km2,
                                   lons2d=lons2d, lats2d=lats2d)

    if stations is not None:
        #Get the list of the corresponding model points
        station_to_modelpoint = cell_manager.get_model_points_for_stations(
            station_list=stations,
            lake_fraction=lake_fraction,
            drainaige_area_reldiff_limit=drainage_area_reldiff_min)

        station_list = list(station_to_modelpoint.keys())
        station_list.sort(key=lambda st1: st1.latitude, reverse=True)
        mp_list = [station_to_modelpoint[st] for st in station_list]
    else:
        mp_list = model_points
        station_list = None
        #sort so that the northernmost stations appear uppermost
        mp_list.sort(key=lambda mpt: mpt.latitude, reverse=True)


    #set ids to the model points so they can be distinguished easier
    model_point.set_model_point_ids(mp_list)



    # brewer2mpl.get_map args: set name  set type  number of colors
    bmap = brewer2mpl.get_map("Set1", "qualitative", 9)
    # Change the default colors
    mpl.rcParams["axes.color_cycle"] = bmap.mpl_colors





    #For the streamflow only plot
    ncols = 3
    nrows = max(len(mp_list) // ncols, 1)
    if ncols * nrows < len(mp_list):
        nrows += 1

    figure_stfl = plt.figure()
    gs_stfl = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    #  a flag which signifies if a legend should be added to the plot, it is needed so we ahve only one legend per plot
    legend_added = False


    ax_stfl = None
    all_years = [y for y in range(start_year, end_year + 1)]

    if station_list is not None:
        processed_stations = station_list
    else:
        processed_stations = [None] * len(mp_list)
    processed_model_points = mp_list
    plot_point_positions_with_upstream_areas(processed_stations, processed_model_points, basemap, cell_manager)




    #create obs data managers
    anusplin_tmin = AnuSplinManager(variable="stmn")
    anusplin_tmax = AnuSplinManager(variable="stmx")
    anusplin_pcp = AnuSplinManager(variable="pcp")

    daily_dates, obs_tmin_fields = anusplin_tmin.get_daily_clim_fields_interpolated_to(
        start_year=start_year, end_year=end_year,
        lons_target=lons2d, lats_target=lats2d)

    _, obs_tmax_fields = anusplin_tmax.get_daily_clim_fields_interpolated_to(
        start_year=start_year, end_year=end_year,
        lons_target=lons2d, lats_target=lats2d)

    _, obs_pcp_fields = anusplin_pcp.get_daily_clim_fields_interpolated_to(
        start_year=start_year, end_year=end_year,
        lons_target=lons2d, lats_target=lats2d)

    swe_manager = SweDataManager(var_name="SWE")
    obs_swe_daily_clim = swe_manager.get_daily_climatology(start_year, end_year)
    interpolated_obs_swe_clim = swe_manager.interpolate_daily_climatology_to(obs_swe_daily_clim,
                                                                             lons2d_target=lons2d,
                                                                             lats2d_target=lats2d)
    values_obs = None
    for i, the_model_point in enumerate(mp_list):

        ax_stfl = figure_stfl.add_subplot(gs_stfl[i // ncols, i % ncols],
                                          sharex=ax_stfl)

        assert isinstance(the_model_point, ModelPoint)

        ##Check the number of years accessible for the station if the list of stations is given
        the_station = None if station_list is None else station_list[i]
        if the_station is not None:
            assert isinstance(the_station, Station)
            year_list = the_station.get_list_of_complete_years()
            year_list = list(itertools.ifilter(lambda yi: start_year <= yi <= end_year, year_list))

            if len(year_list) < 1:
                continue
        else:
            year_list = all_years

        fig = plt.figure(figsize=(12, 15))

        gs = gridspec.GridSpec(4, 4, wspace=1)


        #plot station position
        ax = fig.add_subplot(gs[3, 0:2])
        upstream_mask = _plot_station_position(ax, the_station, basemap, cell_manager, the_model_point)



        #plot streamflows
        ax = fig.add_subplot(gs[0:2, 0:2])

        dates = None
        model_daily_temp_clim = {}
        model_daily_precip_clim = {}
        model_daily_clim_surf_runoff = {}
        model_daily_clim_subsurf_runoff = {}
        model_daily_clim_swe = {}

        #get model data for the list of years
        simlabel_to_vals = {}
        for label in label_list:
            fname = sim_name_to_file_name[label]
            fpath = os.path.join(hdf_folder, fname)
            #read temperature data and calculate daily climatologic fileds
            _, model_daily_temp_clim[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="TT", level=1, start_year=start_year, end_year=end_year)

            #read modelled precip and calculate daily climatologic fields
            _, model_daily_precip_clim[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="PR", level=None, start_year=start_year, end_year=end_year)

            #read modelled surface runoff and calculate daily climatologic fields
            _, model_daily_clim_surf_runoff[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="TRAF", level=1, start_year=start_year, end_year=end_year)

            #read modelled subsurface runoff and calculate daily climatologic fields
            _, model_daily_clim_subsurf_runoff[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="TDRA", level=1, start_year=start_year, end_year=end_year)

            #read modelled swe and calculate daily climatologic fields
            _, model_daily_clim_swe[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="I5", level=None, start_year=start_year, end_year=end_year)

            dates, values_model = analysis.get_daily_climatology_for_a_point(path=fpath,
                                                                             var_name=stfl_name,
                                                                             years_of_interest=year_list,
                                                                             i_index=the_model_point.ix,
                                                                             j_index=the_model_point.jy)

            ax.plot(dates, values_model, label=label, lw=2)
            ax_stfl.plot(dates, values_model, label=label, lw=2)
            simlabel_to_vals[label] = values_model


        if the_station is not None:
            assert isinstance(the_station, Station)
            dates, values_obs = the_station.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=dates,
                                                                                                 years=year_list)

            #To keep the colors consistent for all the variables, the obs Should be plotted last
            ax.plot(dates, values_obs, label="Obs.", lw=2)
            #no ticklabels for streamflow plot
            plt.setp(ax.get_xticklabels(), visible=False)
            ax_stfl.plot(dates, values_obs, label="Obs.", lw=2)

            #Print excesss from streamflow validation
            for label, values_model in simlabel_to_vals.iteritems():
                calclulate_spring_peak_err(dates, values_obs, values_model,
                                           st_id="{}: {}".format(label, the_station.id),
                                           da_mod=the_model_point.accumulation_area,
                                           da_obs=the_station.drainage_km2)


        ax.set_ylabel(r"Streamflow: ${\rm m^3/s}$")
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)

        upstream_area_km2 = np.sum(cell_area_km2[upstream_mask == 1])
        #Put some information about the point
        if the_station is not None:
            lf_upstream = lake_fraction[upstream_mask == 1]
            point_info = "{0}\nlf-max={1:.2f}".format(the_station.id, lf_upstream.max())
            write_annual_flows_to_txt(label_list, simlabel_to_vals, values_obs, file_annual_discharge,
                                      station_id = the_station.id,
                                      da_obs = the_station.drainage_km2, da_mod = the_model_point.accumulation_area)

        else:
            point_info = "{0}".format(the_model_point.point_id)

        ax.text(0.6, 0.9, point_info, transform=ax.transAxes, bbox=dict(facecolor="white"))
        ax_stfl.text(0.6, 0.85, point_info, transform=ax_stfl.transAxes, bbox=dict(facecolor="white"), fontsize=8)

        ax.legend(loc=(0.0, 1.05), borderaxespad=0, ncol=3)
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1, 13, 2)))
        ax.grid()
        streamflow_axes = ax  # save streamflow axes for later use

        if not legend_added:
            ax_stfl.legend(loc=(0.0, 1.05), borderaxespad=0, ncol=3)
            ax_stfl.xaxis.set_major_formatter(DateFormatter("%b"))
            ax_stfl.xaxis.set_minor_locator(MonthLocator())
            ax_stfl.xaxis.set_major_locator(MonthLocator(bymonth=range(1, 13, 3)))
            ax_stfl.set_ylabel(r"Streamflow ${\rm m^3/s}$")
            legend_added = True

        ax_stfl.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_stfl.grid()




        #plot temperature comparisons (tmod - daily with anusplin tmin and tmax)
        ax = fig.add_subplot(gs[3, 2:], sharex=streamflow_axes)
        _validate_temperature_with_anusplin(ax, the_model_point, cell_area_km2=cell_area_km2,
                                            upstream_mask=upstream_mask,
                                            daily_dates=daily_dates,
                                            obs_tmin_clim_fields=obs_tmin_fields,
                                            obs_tmax_clim_fields=obs_tmax_fields,
                                            model_data_dict=model_daily_temp_clim,
                                            simlabel_list=label_list)

        #plot temperature comparisons (tmod - daily with anusplin tmin and tmax)
        ax = fig.add_subplot(gs[2, 2:], sharex=streamflow_axes)
        _validate_precip_with_anusplin(ax, the_model_point, cell_area_km2=cell_area_km2,
                                       upstream_mask=upstream_mask,
                                       daily_dates=daily_dates,
                                       obs_precip_clim_fields=obs_pcp_fields,
                                       model_data_dict=model_daily_precip_clim,
                                       simlabel_list=label_list)


        #plot mean upstream surface runoff
        ax = fig.add_subplot(gs[0, 2:], sharex=streamflow_axes)
        _plot_upstream_surface_runoff(ax, the_model_point, cell_area_km2=cell_area_km2,
                                      upstream_mask=upstream_mask,
                                      daily_dates=daily_dates,
                                      model_data_dict=model_daily_clim_surf_runoff,
                                      simlabel_list=label_list)


        #plot mean upstream subsurface runoff
        ax = fig.add_subplot(gs[1, 2:], sharex=streamflow_axes, sharey=ax)
        _plot_upstream_subsurface_runoff(ax, the_model_point, cell_area_km2=cell_area_km2,
                                         upstream_mask=upstream_mask,
                                         daily_dates=daily_dates,
                                         model_data_dict=model_daily_clim_subsurf_runoff,
                                         simlabel_list=label_list)

        #plot mean upstream swe comparison
        ax = fig.add_subplot(gs[2, 0:2], sharex=streamflow_axes)
        print "Validating SWE for ", the_station.id, "--" * 20
        _validate_swe_with_ross_brown(ax, the_model_point, cell_area_km2=cell_area_km2,
                                      upstream_mask=upstream_mask,
                                      daily_dates=daily_dates,
                                      model_data_dict=model_daily_clim_swe,
                                      obs_swe_clim_fields=interpolated_obs_swe_clim,
                                      simlabel_list=label_list)

        if the_station is not None:
            im_name = "comp_point_with_obs_{0}_{1}_{2}.jpeg".format(the_station.id,
                                                                    the_station.source,
                                                                    "_".join(label_list))
            im_folder_path = os.path.join(images_folder, the_station.source)
        else:
            im_name = "comp_point_with_obs_{0}_{1}.jpeg".format(the_model_point.point_id,
                                                                "_".join(label_list))
            im_folder_path = os.path.join(images_folder, "outlets_point_comp")


        #create a folder for a given source of observed streamflow if it does not exist yet
        if not os.path.isdir(im_folder_path):
            os.mkdir(im_folder_path)

        im_path = os.path.join(im_folder_path, im_name)

        fig.savefig(im_path, dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)


        #return  # temporary plot only one point

    assert isinstance(figure_stfl, Figure)
    figure_stfl.tight_layout()
    figure_stfl.savefig(os.path.join(images_folder, "comp_point_with_obs_{0}.jpeg".format("_".join(label_list))),
                        dpi=cpp.FIG_SAVE_DPI,
                        bbox_inches="tight")
    plt.close(figure_stfl)

    #close information text files
    for f in text_files:
        f.close()


def plot_point_positions_with_upstream_areas(processed_stations, processed_model_points,
                                             basemap, cell_manager):
    #plot point positions with upstream areas
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_positions_of_station_list(ax, processed_stations, processed_model_points, basemap, cell_manager)
    impath = os.path.join(images_folder, "station_positions.jpeg")
    fig.savefig(impath, dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def point_comparisons_at_outlets(hdf_folder="/home/huziy/skynet3_rech1/hdf_store"):
    start_year = 1979
    end_year = 1981

    sim_name_to_file_name = {
        #"CRCM5-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf",
        #"CRCM5-HCD-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf",
        "CRCM5-HCD-RL": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf",
        "CRCM5-HCD-RL-INTFL": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf",
        #"SANI=10000, ignore THFC":
        #    "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_sani-10000_not_care_about_thfc.hdf",

        #"CRCM5-HCD-RL-ERA075": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap_era075.hdf",
        "SANI=10000": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_sani-10000.hdf"
        #"CRCM5-HCD-RL-ECOCLIMAP": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap.hdf"
    }

    path0 = os.path.join(hdf_folder, sim_name_to_file_name.items()[0][1])
    path1 = os.path.join(hdf_folder, sim_name_to_file_name.items()[1][1])
    flow_directions = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_FLOW_DIRECTIONS_NAME)
    lake_fraction = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_LAKE_FRACTION_NAME)
    slope = analysis.get_array_from_file(path=path1, var_name=infovar.HDF_SLOPE_NAME)

    lons2d, lats2d, _ = analysis.get_basemap_from_hdf(file_path=path0)

    cell_manager = CellManager(flow_directions, lons2d=lons2d, lats2d=lats2d)
    mp_list = cell_manager.get_model_points_of_outlets(lower_accumulation_index_limit=10)

    assert len(mp_list) > 0

    #Get the accumulation indices so that the most important outlets can be identified
    acc_ind_list = [np.sum(cell_manager.get_mask_of_cells_connected_with_by_indices(mp.ix, mp.jy))
                    for mp in mp_list]

    for mp, acc_ind in zip(mp_list, acc_ind_list):
        mp.acc_index = acc_ind

    mp_list.sort(key=lambda x: x.acc_index)

    #do not take global lake cells into consideration, and discard points with slopes 0 or less
    mp_list = [mp for mp in mp_list if lake_fraction[mp.ix, mp.jy] < 0.6 and slope[mp.ix, mp.jy] >= 0]

    mp_list = mp_list[-12:]  # get 12 most important outlets

    print "The following outlets were chosen for analysis"
    pattern = "({0}, {1}): acc_index = {2} cells; fldr = {3}; lake_fraction = {4}"
    for mp in mp_list:
        print pattern.format(mp.ix, mp.jy, mp.acc_index, cell_manager.flow_directions[mp.ix, mp.jy],
                             lake_fraction[mp.ix, mp.jy])

    draw_model_comparison(model_points=mp_list, sim_name_to_file_name=sim_name_to_file_name, hdf_folder=hdf_folder,
                          start_year=start_year, end_year=end_year, cell_manager=cell_manager)


def main(hdf_folder="/home/huziy/skynet3_rech1/hdf_store", start_date=None, end_date=None,
         min_station_accumulation_area_km2=1000.0):
    # Station ids to get from the CEHQ database
    # selected_ids = ["092715", "080101", "074903", "050304", "080104", "081007", "061905",
    #                 "041903", "040830", "093806", "090613", "081002", "093801", "080718"]

    ids_with_lakes_upstream = [
        "104001", "093806", "093801", "081002", "081007", "080718"
    ]

    selected_ids = ["092715", "074903", "080104", "081007", "061905",
                    "093806", "090613", "081002", "093801", "080718", "104001"]

    selected_ids = ids_with_lakes_upstream
    #selected_ids = []  # Do not use CEHQ stations temporarily

    # selected_ids = [
    #     "074903", "061905", "090613", "092715", "093801", "093806", "081002"
    # ]

    #selected_ids = ["081002", "104001"]



    #selected_ids = ["090613", ]

    sim_labels = [
        #"CRCM5-R", "CRCM5-HCD-R",
        "Intf1", "Intf-avoid-trunc-higher-slp"
    ]

    sim_file_names = [
        #"quebec_0.1_crcm5-r.hdf5",
        #"quebec_0.1_crcm5-hcd-r.hdf5",
        "quebec_0.1_crcm5-hcd-rl-intfl_ITFS.hdf5",
        "quebec_0.1_crcm5-hcd-rl-intfl_ITFS_avoid_truncation1979-1989.hdf5"

    ]

    sim_name_to_file_name = OrderedDict()
    for k, v in zip(sim_labels, sim_file_names):
        sim_name_to_file_name[k] = v

        #sim_name_to_file_name = {
        #"CRCM5-R": "quebec_0.1_crcm5-r.hdf5",
        #"CRCM5-HCD-R": "quebec_0.1_crcm5-hcd-r.hdf5",
        #"CRCM5-HCD-RL": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf",
        #"CRCM5-HCD-RL-INTFL": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf",
        #"SANI=10000, ignore THFC":
        #    "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_sani-10000_not_care_about_thfc.hdf",

        #"CRCM5-HCD-RL-ERA075": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap_era075.hdf",
        #"SANI=10000": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_sani-10000.hdf"
        #"CRCM5-HCD-RL-ECOCLIMAP": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap.hdf"
    #}





    #Get the list of stations to do the comparison with
    stations = cehq_station.read_station_data(
        start_date=start_date, end_date=end_date, selected_ids=selected_ids
    )

    print "Initial list of stations:"
    for s in stations:
        print u"{0}".format(s)


    #Commented hydat station for performance during testing
    # province = "QC"
    # selected_ids_hydat = None
    # stations_hd = cehq_station.load_from_hydat_db(start_date=start_date, end_date=end_date,
    #                                               province=province,
    #                                               min_drainage_area_km2=min_station_accumulation_area_km2,
    #                                               selected_ids=selected_ids_hydat)
    # if not len(stations_hd):
    #     print "No hydat stations satisying the conditions: period {0}-{1}, province {2}".format(
    #         str(start_date), str(end_date), province
    #     )
    #
    # stations.extend(stations_hd)
    #
    # province = "ON"
    # stations_hd = cehq_station.load_from_hydat_db(start_date=start_date, end_date=end_date,
    #                                               province=province,
    #                                               min_drainage_area_km2=min_station_accumulation_area_km2)
    # stations.extend(stations_hd)

    draw_model_comparison(model_points=None, sim_name_to_file_name=sim_name_to_file_name,
                          hdf_folder=hdf_folder,
                          start_year=start_date.year, end_year=end_date.year, stations=stations,
                          stfl_name="STFA",
                          drainage_area_reldiff_min=0.1)


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()

    main()