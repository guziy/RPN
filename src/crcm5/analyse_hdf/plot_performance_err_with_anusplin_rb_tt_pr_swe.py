from data import cehq_station

from crcm5 import infovar
from crcm5.analyse_hdf.climate_change.plot_performance_err_with_cru import plot_seasonal_mean_biases
from data.anusplin import AnuSplinManager
from data.cell_manager import CellManager
from data.swe import SweDataManager
from util.plot_utils import draw_upstream_area_bounds


__author__ = 'huziy'


# This is done in crcm5/analyse_hdf/common_plotter_hdf_crcm5.py (it calls the right script)
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.basemap import maskoceans
from crcm5.analyse_hdf.run_config import RunConfig
from util import plot_utils
from util.seasons_info import DEFAULT_SEASON_TO_MONTHS
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import numpy as np

# Plot structural and boundary forcing errors for all seasons in the same figure

img_folder = Path("images_for_lake-river_paper/perf_err_with_anusplin_rb_tt_pr_swe")


def compare_vars(vname_model, vname_to_obs, r_config, season_to_months, bmp_info_agg, axes_list):
    season_to_clim_fields_model = analysis.get_seasonal_climatology_for_runconfig(run_config=r_config,
                                                                                  varname=vname_model, level=0,
                                                                                  season_to_months=season_to_months)

    for season, field in season_to_clim_fields_model.items():
        print(field.shape)
        if vname_model == "PR":
            field *= 1.0e3 * 24 * 3600

    seasonal_clim_fields_obs = vname_to_obs[vname_model]

    lons = bmp_info_agg.lons.copy()
    lons[lons > 180] -= 360

    season_to_err = OrderedDict()
    for season in seasonal_clim_fields_obs:
        season_to_err[season] = season_to_clim_fields_model[season] - seasonal_clim_fields_obs[season]
        season_to_err[season] = maskoceans(lons, bmp_info_agg.lats, season_to_err[season], inlands=False)

    cs = plot_seasonal_mean_biases(season_to_error_field=season_to_err,
                                   varname=vname_model,
                                   basemap_info=bmp_info_agg,
                                   axes_list=axes_list)

    return cs


def get_seasonal_clim_obs_data(rconfig=None, vname="TT", bmp_info=None, season_to_months=None, obs_path=None):
    """

    :param rconfig:
    :param vname: Corresponding model variable name i.e either TT or PR
    """
    assert isinstance(rconfig, RunConfig)

    if season_to_months is None:
        season_to_months = DEFAULT_SEASON_TO_MONTHS

    if bmp_info is None:
        bmp_info = analysis.get_basemap_info_from_hdf(file_path=rconfig.data_path)

    obs_query_params = dict(
        start_year=rconfig.start_year,
        end_year=rconfig.end_year,
        lons_target=bmp_info.lons,
        lats_target=bmp_info.lats
    )

    # Calculate daily mean temperatures as T = (T(min) + T(max)) * 0.5
    if vname == "TT":
        tmax_obs_manager = AnuSplinManager(variable="stmx", folder_path=obs_path)
        tmin_obs_manager = AnuSplinManager(variable="stmn", folder_path=obs_path)

        dates, vals_max = tmax_obs_manager.get_daily_clim_fields_interpolated_to(**obs_query_params)

        _, vals_min = tmin_obs_manager.get_daily_clim_fields_interpolated_to(**obs_query_params)

        daily_obs = (dates, (vals_min + vals_max) * 0.5)

    elif vname == "PR":
        pcp_obs_manager = AnuSplinManager(variable="pcp", folder_path=obs_path)
        daily_obs = pcp_obs_manager.get_daily_clim_fields_interpolated_to(**obs_query_params)

    # SWE
    elif vname == "I5":
        swe_manager = SweDataManager(var_name="SWE", path=obs_path)
        daily_obs = swe_manager.get_daily_clim_fields_interpolated_to(**obs_query_params)

    else:
        raise Exception("Unknown variable: {}".format(vname))

    season_to_obs_data = OrderedDict()
    for season, months in season_to_months.items():
        season_to_obs_data[season] = np.mean([f for d, f in zip(*daily_obs) if d.month in months], axis=0)

    return season_to_obs_data


def main():
    season_to_months = DEFAULT_SEASON_TO_MONTHS

    r_config = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r.hdf5",
        start_year=1990, end_year=2010, label="CRCM5-L"
    )

    bmp_info = analysis.get_basemap_info_from_hdf(file_path=r_config.data_path)
    bmp_info.should_draw_grey_map_background = True
    bmp_info.should_draw_basin_boundaries = False
    bmp_info.map_bg_color = "0.75"

    station_ids = [
        "104001", "093806", "093801", "081002", "081007", "080718"
    ]

    # get river network information used in the model
    flow_directions = analysis.get_array_from_file(r_config.data_path, var_name=infovar.HDF_FLOW_DIRECTIONS_NAME)
    accumulation_area_km2 = analysis.get_array_from_file(path=r_config.data_path,
                                                         var_name=infovar.HDF_ACCUMULATION_AREA_NAME)
    cell_manager = CellManager(flow_dirs=flow_directions,
                               lons2d=bmp_info.lons,
                               lats2d=bmp_info.lats, accumulation_area_km2=accumulation_area_km2)

    # Get the list of stations to indicate on the bias map
    stations = cehq_station.read_station_data(
        start_date=None, end_date=None, selected_ids=station_ids
    )
    """:type : list[Station]"""

    xx, yy = bmp_info.get_proj_xy()
    station_to_modelpoint = cell_manager.get_model_points_for_stations(station_list=stations)
    upstream_edges = cell_manager.get_upstream_polygons_for_points(
        model_point_list=station_to_modelpoint.values(), xx=xx, yy=yy)


    # Validate temperature, precip and swe
    obs_path_anusplin = "/home/huziy/skynet3_rech1/anusplin_links"
    obs_path_swe = "data/swe_ross_brown/swe.nc"
    model_var_to_obs_path = OrderedDict([
        ("TT", obs_path_anusplin),
        ("PR", obs_path_anusplin),
        ("I5", obs_path_swe)
    ])

    vname_to_obs_data = {}

    # parameters that won't change in the loop over variable names
    params_const = dict(rconfig=r_config, bmp_info=bmp_info, season_to_months=season_to_months)

    for vname, obs_path in model_var_to_obs_path.items():
        season_to_obs_data = get_seasonal_clim_obs_data(vname=vname, obs_path=obs_path, **params_const)

        # Comment swe over lakes, since I5 calculated only for land
        if vname in ["I5", ]:
            for season in season_to_obs_data:
                season_to_obs_data[season] = maskoceans(bmp_info.lons, bmp_info.lats,
                                                        season_to_obs_data[season],
                                                        inlands=True)

        vname_to_obs_data[vname] = season_to_obs_data


    # Plotting
    plot_all_vars_in_one_fig = True

    fig = None
    gs = None
    row_axes = []
    ncols = None
    if plot_all_vars_in_one_fig:
        plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=20)
        fig = plt.figure()
        ncols = len(season_to_months) + 1
        gs = GridSpec(len(model_var_to_obs_path), ncols, width_ratios=(ncols - 1) * [1., ] + [0.05, ])
    else:
        plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=25)

    row = 0
    station_x_list = []
    station_y_list = []
    for mname in model_var_to_obs_path:

        if plot_all_vars_in_one_fig:
            row_axes = [fig.add_subplot(gs[row, col]) for col in range(ncols)]

        compare_vars(vname_model=mname, vname_to_obs=vname_to_obs_data,
                     r_config=r_config,
                     season_to_months=season_to_months,
                     bmp_info_agg=bmp_info,
                     axes_list=row_axes)

        # -1 in order to exclude colorbars
        for the_ax in row_axes[:-1]:

            # Need titles only for the first row
            if row > 0:
                the_ax.set_title("")

            draw_upstream_area_bounds(the_ax, upstream_edges)

            if len(station_x_list) == 0:
                for the_station in stations:
                    xst, yst = bmp_info.basemap(the_station.longitude, the_station.latitude)
                    station_x_list.append(xst)
                    station_y_list.append(yst)

            bmp_info.basemap.scatter(station_x_list, station_y_list, c="g", ax=the_ax, s=5, zorder=10, alpha=0.5)



        # Hide fall swe
        if mname in ["I5"]:
            row_axes[-2].set_visible(False)

        row += 1


    # Save the figure if necessary
    if plot_all_vars_in_one_fig:
        fig_path = img_folder.joinpath("{}.png".format("_".join(model_var_to_obs_path)))
        with fig_path.open("wb") as figfile:
            fig.savefig(figfile, format="png", bbox_inches="tight")

        plt.close(fig)


def main_wrapper():
    import application_properties

    application_properties.set_current_directory()

    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    main()


if __name__ == '__main__':
    main_wrapper()


