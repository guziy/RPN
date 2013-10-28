import copy
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
from mpltools import color
import pandas
from pandas.tseries.converter import _daily_finder
from crcm5 import infovar
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


def _plot_station_position(ax, the_station, basemap, cell_manager, model_point):
    assert isinstance(the_station, Station)
    assert isinstance(model_point, ModelPoint)
    assert isinstance(cell_manager, CellManager)

    x, y = basemap(cell_manager.lons2d, cell_manager.lats2d)


    #plot the arrows for upstream cells
    ups_mask = cell_manager.get_mask_of_cells_connected_with_by_indices(model_point.ix, model_point.jy)

    x1d_start = x[ups_mask == 1]
    y1d_start = y[ups_mask == 1]
    fld1d = cell_manager.flow_directions[ups_mask == 1]
    i_upstream, j_upstream = np.where(ups_mask == 1)

    imin, jmin = i_upstream.min() - 40, j_upstream.min() - 40
    imax, jmax = imin + 80, jmin + 80

    basemap.llcrnrx = x[imin, jmin]
    basemap.llcrnry = y[imin, jmin]
    basemap.urcrnrx = x[imax, jmax]
    basemap.urcrnry = y[imax, jmax]

    x_station, y_station = basemap(the_station.longitude, the_station.latitude)
    basemap.scatter(x_station, y_station, c="b", s=50, ax=ax, linewidths=0, zorder=2)

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
    return ups_mask


#noinspection PyNoneFunctionAssignment
def _validate_temperature_with_anusplin(ax, model_point, model_data_dict=None,
                                        obs_tmin_clim_fields=None,
                                        obs_tmax_clim_fields=None, daily_dates=None,
                                        cell_area_km2=None,
                                        upstream_mask=None,
                                        simlabel_list=None):
    # model_data_dict - is a dictionary {sim. label: daily climatological fields}
    # obs_tmin_clim_fields - daily climatological fields
    # The order of plotting follows simlabel_list and the observations are plotted the last

    assert isinstance(ax, Axes)
    assert isinstance(model_point, ModelPoint)

    basin_area_km2 = np.tensordot(cell_area_km2, upstream_mask)
    area_matrix = upstream_mask * cell_area_km2

    obs_tmax_clim_fields[np.isnan(obs_tmax_clim_fields)] = -9999
    obs_tmin_clim_fields[np.isnan(obs_tmin_clim_fields)] = -9999

    basin_tmax = np.tensordot(obs_tmax_clim_fields, area_matrix) / basin_area_km2
    basin_tmin = np.tensordot(obs_tmin_clim_fields, area_matrix) / basin_area_km2

    ax.set_ylabel(r"2m temperature: ${\rm ^\circ C}$")

    resample_period = "5D"  # 5 day running mean

    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2
        df = _apply_running_mean(daily_dates, data)
        ax.plot(df.index, df["values"], label=label, lw=1)

    #plot max temperature
    df_tmax = _apply_running_mean(daily_dates, basin_tmax)
    print df_tmax.tail(20)
    p = ax.plot(df_tmax.index, df_tmax["values"], label="tmax-obs", lw=1)

    #plot min temperature
    df_tmin = _apply_running_mean(daily_dates, basin_tmin)
    print df_tmin.tail(20)
    print "---"
    print df.tail(20)
    ax.plot(df_tmin.index, df_tmin["values"], label="tmin-obs", color=p[0].get_color(), lw=1)
    ax.fill_between(df_tmin.index, df_tmax["values"], df_tmin["values"], alpha=0.3, color=p[0].get_color())

    print "--" * 20
    print "tmin ranges: from {0} to {1}".format(np.min(basin_tmin), np.max(basin_tmin))
    print "tmin ranges: from {0} to {1}".format(np.min(df_tmin["values"]), np.max(df_tmin["values"]))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_label_position("right")
    ax.grid()


#noinspection PyNoneFunctionAssignment
def _validate_precip_with_anusplin(ax, model_point, model_data_dict=None,
                                   obs_precip_clim_fields=None,
                                   daily_dates=None,
                                   cell_area_km2=None,
                                   upstream_mask=None,
                                   simlabel_list=None):
    # model_data_dict - is a dictionary {sim. label: daily climatological fields}
    # obs_tmin_clim_fields - daily climatological fields
    # The order of plotting follows simlabel_list and the observations are plotted the last

    assert isinstance(ax, Axes)
    assert isinstance(model_point, ModelPoint)

    basin_area_km2 = np.tensordot(cell_area_km2, upstream_mask)
    area_matrix = upstream_mask * cell_area_km2

    obs_precip_clim_fields[np.isnan(obs_precip_clim_fields)] = -9999

    basin_precip = np.tensordot(obs_precip_clim_fields, area_matrix) / basin_area_km2

    ax.set_ylabel(r"Total precip.: mm/day")

    resample_period = "5D"

    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2

        #convert m/s to mm/day
        data = data * 1000.0 * 24 * 60 * 60

        df = _apply_running_mean(daily_dates, data)

        ax.plot(df.index, df["values"], label=label, lw=1)


    #running mean
    df = _apply_running_mean(daily_dates, basin_precip)

    ax.plot(df.index, df["values"], label="precip-obs", lw=2)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_label_position("right")
    ax.grid()


#noinspection PyNoneFunctionAssignment
def _validate_swe_with_ross_brown(ax, model_point, model_data_dict=None,
                                  obs_swe_clim_fields=None,
                                  daily_dates=None,
                                  cell_area_km2=None,
                                  upstream_mask=None,
                                  simlabel_list=None):
    # model_data_dict - is a dictionary {sim. label: daily climatological fields}
    # obs_tmin_clim_fields - daily climatological fields
    # The order of plotting follows simlabel_list and the observations are plotted the last

    assert isinstance(ax, Axes)
    assert isinstance(model_point, ModelPoint)

    basin_area_km2 = np.tensordot(cell_area_km2, upstream_mask)
    area_matrix = upstream_mask * cell_area_km2

    basin_swe = np.tensordot(obs_swe_clim_fields, area_matrix) / basin_area_km2

    ax.set_ylabel(r"swe: mm")

    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2
        ax.plot(daily_dates, data, label=label, lw=2)

    p = ax.plot(daily_dates, basin_swe, label="swe-obs", lw=2)


    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid()


def _apply_running_mean(index, values, averaging_period = "5D"):
    """

    :param index:
    :param values:
    :param averaging_period:
    :return: df - resampled pandas.DataFrame
    """
    df = pandas.DataFrame(index=index, data=values, columns=["values"])
    df = pandas.rolling_mean(df, window=1, freq=averaging_period)
    return df[:-1]


#noinspection PyNoneFunctionAssignment
def _plot_upstream_surface_runoff(ax, model_point, model_data_dict=None,
                                  daily_dates=None,
                                  cell_area_km2=None,
                                  upstream_mask=None,
                                  simlabel_list=None):
    #plot arae averaged upstream surface runoff

    assert isinstance(ax, Axes)
    assert isinstance(model_point, ModelPoint)

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
    ax.grid()


#noinspection PyNoneFunctionAssignment
def _plot_upstream_subsurface_runoff(ax, model_point, model_data_dict=None,
                                     daily_dates=None,
                                     cell_area_km2=None,
                                     upstream_mask=None,
                                     simlabel_list=None):
    #plot arae averaged upstream surface runoff

    assert isinstance(ax, Axes)
    assert isinstance(model_point, ModelPoint)

    basin_area_km2 = np.tensordot(cell_area_km2, upstream_mask)
    area_matrix = upstream_mask * cell_area_km2

    ax.set_ylabel("Subsurf. r-off: mm/day")
    ax.yaxis.set_label_position("right")

    for label in simlabel_list:
        data = np.tensordot(model_data_dict[label], area_matrix) / basin_area_km2
        #convert mm/s to mm/day
        data = data * 24 * 60 * 60

        df = _apply_running_mean(daily_dates, data)
        ax.plot(df.index, df["values"], label=label, lw=1)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid()


def main(hdf_folder="/home/huziy/skynet3_rech1/hdf_store"):
    import application_properties

    application_properties.set_current_directory()

    start_date = datetime(1979, 1, 1)
    end_date = datetime(1985, 12, 31)

    # Station ids to get from the CEHQ database
    #selected_ids = ["092715", "080101", "074903", "050304", "080104", "081007", "061905",
    #                "041903", "040830", "093806", "090613", "081002", "093801", "080718"]

    selected_ids = [
        "061905", "074903", "090613", "092715", "093801", "093806", "081002"
    ]

    #selected_ids = ["081002", ]

    # brewer2mpl.get_map args: set name  set type  number of colors
    bmap = brewer2mpl.get_map("Set1", "qualitative", 9)
    # Change the default colors
    mpl.rcParams["axes.color_cycle"] = bmap.mpl_colors


    #selected_ids = ["090613", ]

    sim_name_to_file_name = {
        "CRCM5-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r_spinup.hdf",
        "CRCM5-HCD-R": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r_spinup2.hdf",
        "CRCM5-HCD-RL": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl_spinup.hdf",
        "CRCM5-HCD-RL-INTFL": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_do_not_discard_small.hdf",
        #"CRCM5-HCD-RL-ECOCLIMAP": "/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl-intfl_spinup_ecoclimap.hdf"
    }

    #Get the list of stations to do the comparison with
    stations = cehq_station.read_station_data(
        start_date=start_date, end_date=end_date, selected_ids=selected_ids
    )


    #For the streamflow only plot
    figure_stfl, axes_stfl = plt.subplots(3, 2, sharex=True, sharey=True)
    figure_stfl.suptitle(r"Streamflow ${\rm m^3/s}$")
    #  a flag which signifies if a legend should be added to the plot, it is needed so we ahve only one legend per plot
    legend_added = False


    #Commented hydat station for performance during testing
    #province = "QC"
    #stations_hd = cehq_station.load_from_hydat_db(start_date=start_date, end_date=end_date, province=province)
    #if not len(stations_hd):
    #    print "No hydat stations satisying the conditions: period {0}-{1}, province {2}".format(
    #        str(start_date), str(end_date), province
    #    )
    #stations.extend(stations_hd)
    #
    #province = "ON"
    #stations_hd = cehq_station.load_from_hydat_db(start_date=start_date, end_date=end_date, province=province)
    #stations.extend(stations_hd)





    #create obs data managers
    anusplin_tmin = AnuSplinManager(variable="stmn")
    anusplin_tmax = AnuSplinManager(variable="stmx")
    anusplin_pcp = AnuSplinManager(variable="pcp")

    path0 = os.path.join(hdf_folder, sim_name_to_file_name.items()[0][1])
    flow_directions = analysis.get_array_from_file(path=path0, var_name="flow_direction")
    lake_fraction = analysis.get_array_from_file(path=path0, var_name="lake_fraction")
    accumulation_area_km2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_ACCUMULATION_AREA_NAME)
    cell_area_km2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_CELL_AREA_NAME)

    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=path0)

    cell_manager = CellManager(flow_directions, accumulation_area_km2=accumulation_area_km2,
                               lons2d=lons2d, lats2d=lats2d)

    daily_dates, obs_tmin_fields = anusplin_tmin.get_daily_clim_fields_interpolated_to(
        start_year=start_date.year, end_year=end_date.year,
        lons_target=lons2d, lats_target=lats2d
    )

    _, obs_tmax_fields = anusplin_tmax.get_daily_clim_fields_interpolated_to(
        start_year=start_date.year, end_year=end_date.year,
        lons_target=lons2d, lats_target=lats2d
    )

    _, obs_pcp_fileds = anusplin_pcp.get_daily_clim_fields_interpolated_to(
        start_year=start_date.year, end_year=end_date.year,
        lons_target=lons2d, lats_target=lats2d
    )


    swe_manager = SweDataManager(var_name="SWE")
    obs_swe_daily_clim = swe_manager.get_daily_climatology(start_date.year, end_date.year)
    interpolated_obs_swe_clim = swe_manager.interpolate_daily_climatology_to(obs_swe_daily_clim,
                                                                             lons2d_target=lons2d,
                                                                             lats2d_target=lats2d)
    #Get the list of the corresponding model points
    station_to_modelpoint = cell_manager.get_model_points_for_stations(station_list=stations,
                                                                       lake_fraction=lake_fraction)


    #sort so that the northernmost stations appear uppermost
    station_list = list(station_to_modelpoint.keys())
    station_list.sort(key=lambda s: s.latitude, reverse=True)
    ax_stfl_list = list(np.ravel(axes_stfl))

    for the_station, ax_stfl in zip(station_list, ax_stfl_list):
        the_model_point = station_to_modelpoint[the_station]



        assert isinstance(the_station, Station)
        year_list = the_station.get_list_of_complete_years()
        year_list = list(itertools.ifilter(lambda y: start_date.year <= y <= end_date.year, year_list))

        if len(year_list) <= 3:
            continue

        fig = plt.figure()

        gs = gridspec.GridSpec(4, 4, wspace=1)

        #plot streamflows
        ax = fig.add_subplot(gs[0:2, 0:2])

        dates = None
        model_daily_temp_clim = {}
        model_daily_precip_clim = {}
        model_daily_clim_surf_runoff = {}
        model_daily_clim_subsurf_runoff = {}
        model_daily_clim_swe = {}

        label_list = list(sim_name_to_file_name.keys())  # Needed to keep the order the same for all subplots
        #get model data for the list of years
        for label in label_list:
            fname = sim_name_to_file_name[label]
            fpath = os.path.join(hdf_folder, fname)
            #read temperature data and calculate daily climatologic fileds
            _, model_daily_temp_clim[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="TT", level=1, start_year=start_date.year, end_year=end_date.year
            )
            #read modelled precip and calculate daily climatologic fields
            _, model_daily_precip_clim[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="PR", level=None, start_year=start_date.year, end_year=end_date.year
            )

            #read modelled surface runoff and calculate daily climatologic fields
            _, model_daily_clim_surf_runoff[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="TRAF", level=5, start_year=start_date.year, end_year=end_date.year
            )

            #read modelled subsurface runoff and calculate daily climatologic fields
            _, model_daily_clim_subsurf_runoff[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="TDRA", level=5, start_year=start_date.year, end_year=end_date.year
            )

            #read modelled swe and calculate daily climatologic fields
            _, model_daily_clim_swe[label] = analysis.get_daily_climatology(
                path_to_hdf_file=fpath, var_name="I5", level=None, start_year=start_date.year, end_year=end_date.year
            )

            dates, values_model = analysis.get_daily_means_for_a_point(path=fpath,
                                                                       var_name="STFL",
                                                                       years_of_interest=year_list,
                                                                       i_index=the_model_point.ix,
                                                                       j_index=the_model_point.jy)

            ax.plot(dates, values_model, label=label, lw=2)
            ax_stfl.plot(dates, values_model, label=label, lw=2)

        dates, values_obs = the_station.get_daily_climatology_for_complete_years_with_pandas(stamp_dates=dates,
                                                                                             years=year_list)
        #To keep the colors consistent for all the variables, the obs Should be plotted last
        ax.plot(dates, values_obs, label="Obs.", lw=2)
        ax_stfl.plot(dates, values_obs, label="Obs.", lw=2)

        ax.set_ylabel(r"Streamflow: ${\rm m^3/s}$")
        assert isinstance(ax, Axes)
        assert isinstance(fig, Figure)
        ax.text(0.1, 0.9, the_station.id, transform=ax.transAxes, bbox=dict(facecolor="white"))
        ax.legend(loc=(0.0, 1.05), borderaxespad=0, ncol=3)
        ax.xaxis.set_major_formatter(DateFormatter("%b"))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_major_locator(MonthLocator(bymonth=range(1, 13, 2)))
        ax.grid()
        streamflow_axes = ax  # save streamflow axes for later use

        #for the plot containing only streamflows for all stations
        ax_stfl.text(0.1, 0.9, the_station.id, transform=ax_stfl.transAxes, bbox=dict(facecolor="white"))
        ax_stfl.grid()
        if not legend_added:
            ax_stfl.text(0.1, 0.9, the_station.id, transform=ax_stfl.transAxes, bbox=dict(facecolor="white"))
            ax_stfl.legend(loc=(0.0, 1.05), borderaxespad=0, ncol=3)
            ax_stfl.xaxis.set_major_formatter(DateFormatter("%b"))
            ax_stfl.xaxis.set_minor_locator(MonthLocator())
            ax_stfl.xaxis.set_major_locator(MonthLocator(bymonth=range(1, 13, 2)))
            legend_added = True





        #plot station position
        ax = fig.add_subplot(gs[3, 0:2])
        upstream_mask = _plot_station_position(ax, the_station, basemap, cell_manager, the_model_point)



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
                                       obs_precip_clim_fields=obs_pcp_fileds,
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
        ax = fig.add_subplot(gs[1, 2:], sharex=streamflow_axes)
        _plot_upstream_subsurface_runoff(ax, the_model_point, cell_area_km2=cell_area_km2,
                                         upstream_mask=upstream_mask,
                                         daily_dates=daily_dates,
                                         model_data_dict=model_daily_clim_subsurf_runoff,
                                         simlabel_list=label_list)

        #plot mean upstream swe comparison
        ax = fig.add_subplot(gs[2, 0:2], sharex=streamflow_axes)
        _validate_swe_with_ross_brown(ax, the_model_point, cell_area_km2=cell_area_km2,
                                      upstream_mask=upstream_mask,
                                      daily_dates=daily_dates,
                                      model_data_dict=model_daily_clim_swe,
                                      obs_swe_clim_fields=interpolated_obs_swe_clim,
                                      simlabel_list=label_list)







        imName = "comp_point_with_obs_{0}_{1}_{2}.jpeg".format(the_station.id, the_station.source, "_".join(label_list))
        imFolderPath = os.path.join(images_folder, the_station.source)
        #create a folder for a given source of observed streamflow if it does not exist yet
        if not os.path.isdir(imFolderPath):
            os.mkdir(imFolderPath)

        imPath = os.path.join(imFolderPath, imName)

        fig.savefig(imPath, dpi=cpp.FIG_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        #return  # temporary plot only one point
    figure_stfl.savefig(os.path.join(images_folder, "comp_point_with_obs_{0}.jpeg".format("_".join(label_list))),
                        dpi=cpp.FIG_SAVE_DPI,
                        bbox_inches="tight")
    plt.close(figure_stfl)


if __name__ == "__main__":
    main()