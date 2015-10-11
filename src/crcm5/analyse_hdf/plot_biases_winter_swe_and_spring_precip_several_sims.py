from collections import OrderedDict
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import maskoceans
from pathlib import Path
from application_properties import main_decorator
from crcm5 import infovar
from crcm5.analyse_hdf.plot_performance_err_with_anusplin_rb_tt_pr_swe import get_seasonal_clim_obs_data, compare_vars
from crcm5.analyse_hdf.run_config import RunConfig
from data import cehq_station
from data.cell_manager import CellManager
from util import plot_utils
from util.plot_utils import draw_upstream_area_bounds

__author__ = 'huziy'

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import matplotlib.pyplot as plt


img_folder = Path("images_for_lake-river_paper/perf_err_with_anusplin_rb_tt_swe")


@main_decorator
def main():
    # Define the simulations to be validated
    r_config = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-r.hdf5",
        start_year=1990, end_year=2010, label="CRCM5-L1"
    )
    r_config_list = [r_config]

    r_config = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-r.hdf5",
        start_year=1990, end_year=2010, label="CRCM5-NL"
    )
    r_config_list.append(r_config)

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

    bmp_info.draw_colorbar_for_each_subplot = True


    # Validate temperature, precip and swe
    obs_path_anusplin = "/home/huziy/skynet3_rech1/anusplin_links"
    obs_path_swe = "data/swe_ross_brown/swe.nc"
    model_var_to_obs_path = OrderedDict([
        ("TT", obs_path_anusplin),
        ("I5", obs_path_swe)
    ])

    model_var_to_season = OrderedDict([
        ("TT",
         OrderedDict([("Spring", range(3, 6))])),
        ("I5",
         OrderedDict([("Winter", [1, 2, 12])]))
    ])

    vname_to_obs_data = {}

    # parameters that won't change in the loop over variable names
    params_const = dict(rconfig=r_config, bmp_info=bmp_info)

    for vname, obs_path in model_var_to_obs_path.items():
        season_to_obs_data = get_seasonal_clim_obs_data(vname=vname,
                                                        obs_path=obs_path,
                                                        season_to_months=model_var_to_season[vname],
                                                        **params_const)

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
    if plot_all_vars_in_one_fig:
        plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=20)
        fig = plt.figure()
        ncols = len(model_var_to_obs_path) + 1
        gs = GridSpec(len(r_config_list), ncols, width_ratios=(ncols - 1) * [1., ] + [0.05, ])
    else:
        plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=25)

    station_x_list = []
    station_y_list = []

    mvarname_to_cs = {}
    for row, r_config in enumerate(r_config_list):
        for col, mname in enumerate(model_var_to_obs_path):

            row_axes = [fig.add_subplot(gs[row, col]), ]

            mvarname_to_cs[mname] = compare_vars(vname_model=mname, vname_to_obs=vname_to_obs_data,
                                                 r_config=r_config,
                                                 season_to_months=model_var_to_season[mname],
                                                 bmp_info_agg=bmp_info,
                                                 axes_list=row_axes)

            # -1 in order to exclude colorbars
            for the_ax in row_axes:


                the_ax.set_title(the_ax.get_title() + ", {}".format(infovar.get_long_display_label_for_var(mname)))
                # Need titles only for the first row
                if row > 0:
                    the_ax.set_title("")



                if col == 0:
                    the_ax.set_ylabel(r_config.label)
                else:
                    the_ax.set_ylabel("")


                draw_upstream_area_bounds(the_ax, upstream_edges, color="g")

                if len(station_x_list) == 0:
                    for the_station in stations:
                        xst, yst = bmp_info.basemap(the_station.longitude, the_station.latitude)
                        station_x_list.append(xst)
                        station_y_list.append(yst)

                bmp_info.basemap.scatter(station_x_list, station_y_list, c="g", ax=the_ax, s=20, zorder=10, alpha=0.5)






    # Save the figure if necessary
    if plot_all_vars_in_one_fig:

        if not img_folder.is_dir():
            img_folder.mkdir(parents=True)

        fig_path = img_folder.joinpath("{}.png".format("_".join(model_var_to_obs_path)))
        with fig_path.open("wb") as figfile:
            fig.savefig(figfile, format="png", bbox_inches="tight")

        plt.close(fig)


if __name__ == '__main__':
    main()
