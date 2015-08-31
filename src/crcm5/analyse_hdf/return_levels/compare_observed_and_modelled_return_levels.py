from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from data import cehq_station
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from crcm5 import infovar
from crcm5.analyse_hdf.return_levels.extreme_commons import ExtremeProperties
from crcm5.analyse_hdf.run_config import RunConfig
from crcm5.model_point import ModelPoint
from data.cehq_station import Station
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
from data.cell_manager import CellManager
from crcm5.analyse_hdf.return_levels import extreme_commons
from gev_dist import gevfit
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from util import plot_utils

__author__ = 'huziy'

img_folder = Path("cc_paper/return_levels/validation/")


def prepare():
    import application_properties

    application_properties.set_current_directory()

    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    plot_utils.apply_plot_params(font_size=10, width_cm=20, height_cm=18)


def main(hdf_folder="/home/huziy/skynet3_rech1/hdf_store", start_year=1980, end_year=2010):
    prepare()

    all_markers = ["*", "s", "p", "+", "x", "d", "h"]

    excluded = ["white", "w", "aliceblue", "azure"]
    excluded.extend([ci for ci in colors.cnames if "yellow" in ci])

    all_colors = ["k", "b", "r", "g", "m"] + sorted([ci for ci in colors.cnames if ci not in excluded])

    # Station ids to get from the CEHQ database
    ids_with_lakes_upstream = [
        "104001", "093806", "093801", "081002", "081007", "080718"
    ]

    selected_ids = ids_with_lakes_upstream

    filedir = Path(hdf_folder)
    sim_name_to_file_path = OrderedDict([
        # ("CRCM5-LI", filedir.joinpath("quebec_0.1_crcm5-hcd-r.hdf5").as_posix()),

        ("ERAI-CRCM5-L", filedir.joinpath("quebec_0.1_crcm5-hcd-rl.hdf5").as_posix()),

        # ("CanESM2-CRCM5-NL", filedir.joinpath("cc-canesm2-driven/quebec_0.1_crcm5-r-cc-canesm2-1980-2010.hdf5").as_posix()),

        ("CanESM2-CRCM5-L",
         filedir.joinpath("cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5").as_posix()),

        # ("CanESM2-CRCM5-LI", filedir.joinpath("cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-intfl-cc-canesm2-1980-2010.hdf5").as_posix()),


    ])

    obs_label = "Obs."
    labels = [obs_label, ] + list(sim_name_to_file_path.keys())

    label_to_marker = dict(zip(labels, all_markers))
    label_to_color = dict(zip(labels, all_colors))

    # Get the list of stations to do the comparison with
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    stations = cehq_station.read_station_data(
        start_date=start_date, end_date=end_date, selected_ids=selected_ids
    )

    # Get geophysical fields from one of the model simulations
    path0 = list(sim_name_to_file_path.values())[0]
    lons2d, lats2d, basemap = analysis.get_basemap_from_hdf(file_path=path0)
    flow_directions = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_FLOW_DIRECTIONS_NAME)
    lake_fraction = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_LAKE_FRACTION_NAME)

    accumulation_area_km2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_ACCUMULATION_AREA_NAME)
    area_m2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_CELL_AREA_NAME_M2)

    # Try to read cell areas im meters if it is not Ok then try in km2
    if area_m2 is not None:
        cell_area_km2 = area_m2 * 1.0e-6
    else:
        cell_area_km2 = analysis.get_array_from_file(path=path0, var_name=infovar.HDF_CELL_AREA_NAME_KM2)

    # Create a cell manager if it is not provided
    cell_manager = CellManager(flow_directions, accumulation_area_km2=accumulation_area_km2,
                               lons2d=lons2d, lats2d=lats2d)

    # Get the list of the corresponding model points
    station_to_modelpoint = cell_manager.get_model_points_for_stations(
        station_list=stations,
        lake_fraction=lake_fraction,
        drainaige_area_reldiff_limit=0.1)

    # plot_utils.apply_plot_params(font_size=10, width_cm=20, height_cm=18)
    fig = plt.figure()

    ncols = max([len(rp_list) for et, rp_list in ExtremeProperties.extreme_type_to_return_periods.items()])
    nrows = len(ExtremeProperties.extreme_types)
    gs = GridSpec(nrows, ncols)

    ext_type_to_rp_to_ax = OrderedDict()
    ax_with_legend = None

    label_to_ax_to_xdata = {}
    label_to_ax_to_ydata = {}
    for row, ext_type in enumerate(ExtremeProperties.extreme_types):
        ext_type_to_rp_to_ax[ext_type] = OrderedDict()
        for col, rperiod in enumerate(ExtremeProperties.extreme_type_to_return_periods[ext_type]):
            ax = fig.add_subplot(gs[row, col])
            ext_type_to_rp_to_ax[ext_type][rperiod] = ax

            if col == 0:
                ax.set_ylabel(ext_type)

            if row == nrows - 1 and col == ncols - 1:
                ax_with_legend = ax

            # Set axes labels
            if row == nrows - 1:
                ax.set_xlabel("Obs.")

            if col == 0:
                ax.set_ylabel("Model")

            for label in sim_name_to_file_path:

                if label not in label_to_ax_to_xdata:
                    label_to_ax_to_xdata[label] = {ax: []}
                    label_to_ax_to_ydata[label] = {ax: []}
                else:
                    label_to_ax_to_xdata[label][ax] = []
                    label_to_ax_to_ydata[label][ax] = []

            ax.set_xscale("log")
            ax.set_yscale("log")

    print("Initial list of stations:")

    sim_label_to_handle = {}
    for s in stations:
        print("{0}".format(s))
        assert isinstance(s, Station)

        print(len([y for y in s.get_list_of_complete_years() if start_year <= y <= end_year]))
        df_ext_obs = extreme_commons.get_annual_extrema(ts_times=s.dates, ts_vals=s.values,
                                                        start_year=start_year, end_year=end_year)
        mp = station_to_modelpoint[s]

        assert isinstance(mp, ModelPoint)

        years_of_interest = df_ext_obs.index

        label_to_extrema_model = {}



        # label -> ext_type -> [return period -> ret level, return period -> std]
        label_to_return_levels = OrderedDict(
            [(obs_label, OrderedDict())]
        )
        for sim_label, sim_path in sim_name_to_file_path.items():
            label_to_return_levels[sim_label] = OrderedDict()
            label_to_extrema_model[sim_label] = OrderedDict()

        # Calculate the return levels and standard deviations
        for ext_type in ExtremeProperties.extreme_types:

            return_periods = ExtremeProperties.extreme_type_to_return_periods[ext_type]

            # fit GEV distribution and apply non-parametric bootstrap to get std
            label_to_return_levels[obs_label][ext_type] = gevfit.do_gevfit_for_a_point(df_ext_obs[ext_type].values,
                                                                                       extreme_type=ext_type,
                                                                                       return_periods=return_periods)
            return_levels_obs, rl_stds_obs = label_to_return_levels[obs_label][ext_type]

            # get annual extremas for the model output at the points colose to the stations
            for sim_label, sim_path in sim_name_to_file_path.items():
                label_to_return_levels[sim_label] = OrderedDict()

                ext_field = analysis.get_annual_extrema(
                    rconfig=RunConfig(data_path=sim_path, start_year=start_year, end_year=end_year),
                    varname="STFL", months_of_interest=ExtremeProperties.extreme_type_to_month_of_interest[ext_type],
                    n_avg_days=ExtremeProperties.extreme_type_to_n_agv_days[ext_type],
                    high_flow=ext_type == ExtremeProperties.high)

                # Select only those years when obs are available
                ts_data = [v for y, v in zip(range(start_year, end_year + 1), ext_field[:, mp.ix, mp.jy]) if
                           y in years_of_interest]
                ts_data = np.array(ts_data)
                return_levels, rl_stds = gevfit.do_gevfit_for_a_point(ts_data, extreme_type=ext_type,
                                                                      return_periods=return_periods)



                # Do the plotting
                for rp in return_periods:
                    ax = ext_type_to_rp_to_ax[ext_type][rp]
                    ax.set_title("T = {rp}-year".format(rp=rp))

                    # h = ax.errorbar(return_levels_obs[rp], return_levels[rp],
                    # marker=label_to_marker[sim_label], color=label_to_color[sim_label], label=sim_label,
                    #                 xerr=rl_stds_obs[rp] * 1.96, yerr=rl_stds[rp] * 1.96)

                    h = ax.scatter(return_levels_obs[rp], return_levels[rp],
                                   marker=label_to_marker[sim_label], color=label_to_color[sim_label], label=sim_label)




                    # save the data for maybe further calculation of the correlation coefficients
                    label_to_ax_to_xdata[sim_label][ax].append(return_levels_obs[rp])
                    label_to_ax_to_ydata[sim_label][ax].append(return_levels[rp])

                    sim_label_to_handle[sim_label] = h

    sfmt = ScalarFormatter(useMathText=True)
    sfmt.set_powerlimits((-2, 2))
    for et, rp_to_ax in ext_type_to_rp_to_ax.items():
        for rp, ax in rp_to_ax.items():
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x1 = min(xmin, ymin)
            x2 = min(xmax, ymax)
            ax.plot([x1, x2], [x1, x2], "k--")
            # ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            # ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            # ax.xaxis.set_major_formatter(sfmt)
            # ax.yaxis.set_major_formatter(sfmt)

    sim_labels = list(sim_name_to_file_path.keys())
    ax_with_legend.legend([sim_label_to_handle[sl] for sl in sim_labels], sim_labels,
                          bbox_to_anchor=(1, -0.25), borderaxespad=0.0, loc="upper right",
                          ncol=2, scatterpoints=1, numpoints=1)

    # Save the plot
    img_file = "{}.eps".format("_".join(sorted(label_to_marker.keys())))
    img_file = img_folder.joinpath(img_file)

    fig.tight_layout()
    with img_file.open("wb") as f:
        fig.savefig(f, bbox_inches="tight")


if __name__ == "__main__":
    main()
