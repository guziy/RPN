from collections import OrderedDict
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.basemap import maskoceans
from crcm5 import infovar
from crcm5.analyse_hdf.climate_change import plot_performance_err_with_cru
from crcm5.analyse_hdf.climate_change.plot_performance_err_with_cru import compare_vars
from util.array_utils import aggregate_array
from crcm5.analyse_hdf.run_config import RunConfig
from util import plot_utils
from util.seasons_info import DEFAULT_SEASON_TO_MONTHS
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis

__author__ = 'huziy'

# Plot structural and boundary forcing errors for all seasons in the same figure

img_folder = Path("cc_paper/swe_pe_and_bfe")


def plot_swe_bfes(runconfig_rea, runconfig_gcm, vname_model="I5", season_to_months=None,
                  bmp_info=None, axes_list=None):

    seasonal_clim_fields_rea = analysis.get_seasonal_climatology_for_runconfig(run_config=runconfig_rea,
                                                                               varname=vname_model, level=0,
                                                                               season_to_months=season_to_months)

    seasonal_clim_fields_gcm = analysis.get_seasonal_climatology_for_runconfig(run_config=runconfig_gcm,
                                                                               varname=vname_model, level=0,
                                                                               season_to_months=season_to_months)

    lons = bmp_info.lons.copy()
    lons[lons > 180] -= 360

    assert len(seasonal_clim_fields_rea) > 0
    season_to_err = OrderedDict()
    for season, field in seasonal_clim_fields_rea.items():
        rea = field
        gcm = seasonal_clim_fields_gcm[season]

        # Mask oceans and lakes

        season_to_err[season] = maskoceans(lons, bmp_info.lats, gcm - rea)
        assert hasattr(season_to_err[season], "mask")

    plot_performance_err_with_cru.plot_seasonal_mean_biases(season_to_error_field=season_to_err,
                                                            varname=vname_model, basemap_info=bmp_info,
                                                            axes_list=axes_list)



def main():
    season_to_months = OrderedDict([(s, months) for s, months in DEFAULT_SEASON_TO_MONTHS.items()
                                    if s.lower() not in ["summer", "fall"]])

    r_config_01 = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=1980, end_year=1996, label="CRCM5-L"
    )

    r_config_04 = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.4_crcm5-hcd-rl.hdf5",
        start_year=1980, end_year=1996, label="CRCM5-L(0.4)"
    )

    r_config = r_config_04

    # Number of points for aggregation
    nx_agg_model = 1
    ny_agg_model = 1

    nx_agg_obs = 2
    ny_agg_obs = 2




    r_config_cc = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/cc-canesm2-driven/quebec_0.1_crcm5-hcd-rl-cc-canesm2-1980-2010.hdf5",
        start_year=1980, end_year=1996, label="CRCM5-L"
    )


    bmp_info = analysis.get_basemap_info_from_hdf(file_path=r_config.data_path)

    if nx_agg_model * ny_agg_model > 1:
        bmp_info_agg = bmp_info.get_aggregated(nagg_x=nx_agg_model, nagg_y=ny_agg_model)
    else:
        bmp_info_agg = bmp_info

    # Validate temperature and precip
    model_vars = ["I5", ]
    obs_vars = ["SWE", ]
    obs_paths = [
        "/RESCUE/skynet3_rech1/huziy/swe_ross_brown/swe.nc4",
    ]

    plot_all_vars_in_one_fig = True

    fig = None
    gs = None
    row_axes = None
    obs_row_axes = None
    ncols = None
    if plot_all_vars_in_one_fig:
        plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=12, height_cm=18)
        fig = plt.figure()
        ncols = len(season_to_months) + 1
        gs = GridSpec(len(model_vars) * 3, ncols, width_ratios=(ncols - 1) * [1., ] + [0.05, ])

    else:
        plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=25)

    row = 0
    for mname, oname, opath in zip(model_vars, obs_vars, obs_paths):

        if plot_all_vars_in_one_fig:
            # Obs row is for observed values
            obs_row_axes = [fig.add_subplot(gs[row, col]) for col in range(ncols)]
            obs_row_axes[-1].set_title("mm")
            row += 1

            model_row_axes = [fig.add_subplot(gs[row, col]) for col in range(ncols - 1)]
            row += 1

            row_axes = [fig.add_subplot(gs[row, col]) for col in range(ncols)]
            row += 1

        compare_vars(vname_model=mname, vname_obs=oname, r_config=r_config,
                     season_to_months=season_to_months,
                     nx_agg_model=nx_agg_model, ny_agg_model=ny_agg_model,
                     nx_agg_obs=nx_agg_obs, ny_agg_obs=ny_agg_obs,
                     bmp_info_agg=bmp_info_agg, bmp_info_model=bmp_info,
                     obs_path=opath,
                     diff_axes_list=row_axes, obs_axes_list=obs_row_axes, model_axes_list=model_row_axes)

        if plot_all_vars_in_one_fig:
            obs_row_axes[0].set_ylabel("(a)", rotation="horizontal", labelpad=10)
            model_row_axes[0].set_ylabel("(b)", rotation="horizontal", labelpad=10)

            row_axes[0].set_ylabel("(c)", rotation="horizontal", labelpad=10)
            for ax in row_axes[:-1]:
                ax.set_title("")

    # row_axes[0].annotate("(a)", (-0.2, 0.5), font_properties=FontProperties(weight="bold"), xycoords="axes fraction")



    # Plot bfe errs
    # row_axes = [fig.add_subplot(gs[row, col]) for col in range(ncols)]
    # plot_swe_bfes(r_config, r_config_cc, vname_model="I5", season_to_months=season_to_months,
    #               bmp_info=bmp_info, axes_list=row_axes)
    #
    # row_axes[-1].set_visible(False)


    # row_axes[0].annotate("(b)", (0, 1.05), font_properties=FontProperties(weight="bold"), xycoords="axes fraction")
    # row_axes[0].set_ylabel("(b)", rotation="horizontal", labelpad=10)
    # for ax in row_axes[:-1]:
    #     ax.set_title("")


    # Save the figure if necessary
    if plot_all_vars_in_one_fig:

        from crcm5.analyse_hdf import common_plot_params

        fig_path = img_folder.joinpath("{}_{}.png".format("_".join(model_vars), r_config.label))
        with fig_path.open("wb") as figfile:
            fig.savefig(figfile, format="png", bbox_inches="tight", dpi=common_plot_params.FIG_SAVE_DPI)

        plt.close(fig)


def main_wrapper():
    import application_properties

    application_properties.set_current_directory()

    if not img_folder.is_dir():
        img_folder.mkdir(parents=True)

    main()


if __name__ == '__main__':
    main_wrapper()
