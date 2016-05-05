from crcm5.analyse_hdf.climate_change.plot_performance_err_with_cru import plot_seasonal_mean_biases
from data.anusplin import AnuSplinManager

__author__ = 'huziy'


# This is done in crcm5/analyse_hdf/common_plotter_hdf_crcm5.py (it calls the right script)
from collections import OrderedDict
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.basemap import maskoceans
from crcm5.analyse_hdf.climate_change import plot_performance_err_with_cru
from crcm5.analyse_hdf.run_config import RunConfig
from util import plot_utils
from util.seasons_info import DEFAULT_SEASON_TO_MONTHS
from crcm5.analyse_hdf import do_analysis_using_pytables as analysis
import numpy as np

# Plot structural and boundary forcing errors for all seasons in the same figure

img_folder = Path("cc_paper/perf_err_with_anusplin")



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
    print("-------------var: {} (PE with anusplin)---------------------".format(vname_model))
    for season in seasonal_clim_fields_obs:
        seasonal_clim_fields_obs[season] = np.ma.masked_where(np.isnan(seasonal_clim_fields_obs[season]),
                                                                       seasonal_clim_fields_obs[season])

        season_to_err[season] = season_to_clim_fields_model[season] - seasonal_clim_fields_obs[season]
        season_to_err[season] = maskoceans(lons, bmp_info_agg.lats, season_to_err[season], inlands=False)

        season_to_err[season] = np.ma.masked_where(np.isnan(season_to_err[season]), season_to_err[season])


        good_vals = season_to_err[season]
        good_vals = good_vals[~good_vals.mask]


        print("{}: min={}; max={}; avg={}".format(season,
                                                  good_vals.min(),
                                                  good_vals.max(),
                                                  good_vals.mean()))

        print("---------percetages --- anuplin ---")
        print("{}: {} %".format(season, good_vals.mean() / seasonal_clim_fields_obs[season][~season_to_err[season].mask].mean() * 100))


    cs = plot_seasonal_mean_biases(season_to_error_field=season_to_err, varname=vname_model, basemap_info=bmp_info_agg,
                                   axes_list=axes_list)

    return cs


def get_seasonal_clim_obs_data(rconfig=None, vname="TT", bmp_info=None, season_to_months=None):
    """

    :param rconfig:
    :param vname: Corresponding model variable name i.e either TT or PR
    """
    assert isinstance(rconfig, RunConfig)


    if season_to_months is None:
        season_to_months = DEFAULT_SEASON_TO_MONTHS

    if bmp_info is None:
        bmp_info = analysis.get_basemap_info_from_hdf(file_path=rconfig.data_path)


    # Get Anusplin data managers
    obs_path = "/home/huziy/skynet3_rech1/anusplin_links"
    pcp_obs_manager = AnuSplinManager(variable="pcp", folder_path=obs_path)
    tmax_obs_manager = AnuSplinManager(variable="stmx", folder_path=obs_path)
    tmin_obs_manager = AnuSplinManager(variable="stmn", folder_path=obs_path)

    if vname == "TT":
        dates, vals_max = tmax_obs_manager.get_daily_clim_fields_interpolated_to(start_year=rconfig.start_year,
                                                                                 end_year=rconfig.end_year,
                                                                                 lons_target=bmp_info.lons,
                                                                                 lats_target=bmp_info.lats)

        _, vals_min = tmin_obs_manager.get_daily_clim_fields_interpolated_to(start_year=rconfig.start_year,
                                                                             end_year=rconfig.end_year,
                                                                             lons_target=bmp_info.lons,
                                                                             lats_target=bmp_info.lats)

        daily_obs = (dates, (vals_min + vals_max) * 0.5)
    elif vname == "PR":
        daily_obs = pcp_obs_manager.get_daily_clim_fields_interpolated_to(start_year=rconfig.start_year,
                                                                          end_year=rconfig.end_year,
                                                                          lons_target=bmp_info.lons,
                                                                          lats_target=bmp_info.lats)
    else:
        raise Exception("Unknown variable: {}".format(vname))

    season_to_obs_data = OrderedDict()
    for season, months in season_to_months.items():
        season_to_obs_data[season] = np.mean([f for d, f in zip(*daily_obs) if d.month in months], axis=0)


    return season_to_obs_data


def main():
    season_to_months = DEFAULT_SEASON_TO_MONTHS

    r_config = RunConfig(
        data_path="/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5",
        start_year=1980, end_year=2010, label="CRCM5-L"
    )

    bmp_info = analysis.get_basemap_info_from_hdf(file_path=r_config.data_path)



    # Validate temperature and precip
    model_vars = ["TT", "PR"]

    # Get Anusplin data managers
    obs_path = "/home/huziy/skynet3_rech1/anusplin_links"
    pcp_obs_manager = AnuSplinManager(variable="pcp", folder_path=obs_path)
    tmax_obs_manager = AnuSplinManager(variable="stmx", folder_path=obs_path)
    tmin_obs_manager = AnuSplinManager(variable="stmn", folder_path=obs_path)

    vname_to_obs_data = {}

    for vname in model_vars:
        if vname == "TT":
            dates, vals_max = tmax_obs_manager.get_daily_clim_fields_interpolated_to(start_year=r_config.start_year,
                                                                                     end_year=r_config.end_year,
                                                                                     lons_target=bmp_info.lons,
                                                                                     lats_target=bmp_info.lats)

            _, vals_min = tmin_obs_manager.get_daily_clim_fields_interpolated_to(start_year=r_config.start_year,
                                                                                 end_year=r_config.end_year,
                                                                                 lons_target=bmp_info.lons,
                                                                                 lats_target=bmp_info.lats)

            daily_obs = (dates, (vals_min + vals_max) * 0.5)
        elif vname == "PR":
            daily_obs = pcp_obs_manager.get_daily_clim_fields_interpolated_to(start_year=r_config.start_year,
                                                                              end_year=r_config.end_year,
                                                                              lons_target=bmp_info.lons,
                                                                              lats_target=bmp_info.lats)
        else:
            raise Exception("Unknown variable: {}".format(vname))

        season_to_obs_data = OrderedDict()
        for season, months in season_to_months.items():
            season_to_obs_data[season] = np.mean([f for d, f in zip(*daily_obs) if d.month in months], axis=0)

        vname_to_obs_data[vname] = season_to_obs_data

    plot_all_vars_in_one_fig = True

    fig = None
    gs = None
    row_axes = None
    ncols = None
    if plot_all_vars_in_one_fig:
        plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=12)
        fig = plt.figure()
        ncols = len(season_to_months) + 1
        gs = GridSpec(len(model_vars), ncols, width_ratios=(ncols - 1) * [1., ] + [0.05, ])
    else:
        plot_utils.apply_plot_params(font_size=12, width_pt=None, width_cm=25, height_cm=25)

    row = 0
    for mname in model_vars:

        if plot_all_vars_in_one_fig:
            row_axes = [fig.add_subplot(gs[row, col]) for col in range(ncols)]

        compare_vars(vname_model=mname, vname_to_obs=vname_to_obs_data, r_config=r_config,
                     season_to_months=season_to_months,
                     bmp_info_agg=bmp_info,
                     axes_list=row_axes)

        row += 1


    # Save the figure if necessary
    if plot_all_vars_in_one_fig:
        fig_path = img_folder.joinpath("{}.png".format("_".join(model_vars)))
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


