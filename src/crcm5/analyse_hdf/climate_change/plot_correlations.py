# Plot the correlations between modelled temperature and precipitation and Anusplin data
from collections import OrderedDict

from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from application_properties import main_decorator
from crcm5.analyse_hdf.run_config import RunConfig
from data.anusplin import AnuSplinManager

from crcm5.analyse_hdf import do_analysis_using_pytables as analysis


import matplotlib.pyplot as plt
import numpy as np
import os
from util import stat_helpers, plot_utils

img_folder = "cc-paper-comments"


@main_decorator
def main():
    start_year = 1980
    end_year = 2010

    season_to_months = OrderedDict([
        ("Winter", [12, 1, 2]),
        ("Spring", [3, 4, 5]),
        ("Summer", [6, 7, 8]),
        ("Fall", [9, 10, 11]),
    ])

    model_path = "/RESCUE/skynet3_rech1/huziy/hdf_store/quebec_0.1_crcm5-hcd-rl.hdf5"

    sim_config = RunConfig(data_path=model_path, start_year=start_year, end_year=end_year, label="ERAI-CRCM5-L")

    plot_corr_fields_and_std(vname="TT", season_to_months=season_to_months, sim_config=sim_config)
    plot_corr_fields_and_std(vname="PR", season_to_months=season_to_months, sim_config=sim_config)


def plot_corr_fields_and_std(vname="TT", season_to_months=None, sim_config=None):
    """

    :param vname:
    :param season_to_months:
    :param sim_config:
    """

    bmp_info = analysis.get_basemap_info_from_hdf(file_path=sim_config.data_path)

    # Get Anusplin data managers

    obs_path = "/home/huziy/skynet3_rech1/anusplin_links"
    pcp_obs_manager = AnuSplinManager(variable="pcp", folder_path=obs_path)
    tmax_obs_manager = AnuSplinManager(variable="stmx", folder_path=obs_path)
    tmin_obs_manager = AnuSplinManager(variable="stmn", folder_path=obs_path)

    season_to_corr = OrderedDict()
    season_to_area_mean_corr = OrderedDict()
    season_to_area_mean_std_diff = OrderedDict()


    season_to_area_mean_obs = OrderedDict()
    season_to_area_mean_model = OrderedDict()

    season_to_std_diff = OrderedDict()

    season_to_years = OrderedDict()


    # Convert modelled precip from M/s to mm/day
    if vname == "PR":
        model_convert_coef = 24 * 3600 * 1000
    else:
        model_convert_coef = 1


    years = None

    for season, months in season_to_months.items():



        # Obs
        if vname == "TT":
            years, vals_max = tmax_obs_manager.get_seasonal_fields_interpolated_to(start_year=sim_config.start_year,
                                                                                   end_year=sim_config.end_year,
                                                                                   lons_target=bmp_info.lons,
                                                                                   lats_target=bmp_info.lats,
                                                                                   months=months)

            _, vals_min = tmin_obs_manager.get_seasonal_fields_interpolated_to(start_year=sim_config.start_year,
                                                                               end_year=sim_config.end_year,
                                                                               lons_target=bmp_info.lons,
                                                                               lats_target=bmp_info.lats, months=months)

            seasonal_obs = (years, (vals_min + vals_max) * 0.5)

        elif vname == "PR":

            seasonal_obs = pcp_obs_manager.get_seasonal_fields_interpolated_to(start_year=sim_config.start_year,
                                                                               end_year=sim_config.end_year,
                                                                               lons_target=bmp_info.lons,
                                                                               lats_target=bmp_info.lats,
                                                                               months=months)
            years = seasonal_obs[0]
        else:
            raise Exception("Unknown variable: {}".format(vname))


        season_to_years[season] = years

        # Model
        seasonal_model = analysis.get_mean_2d_fields_for_months(path=sim_config.data_path, var_name=vname, level=0,
                                                               months=months, start_year=sim_config.start_year, end_year=sim_config.end_year)


        seasonal_model *= model_convert_coef


        # 2d fields
        season_to_corr[season] = stat_helpers.calculate_correlation_nd(seasonal_obs[1], seasonal_model)
        season_to_corr[season] = np.ma.masked_where(seasonal_obs[1][0, :, :].mask, season_to_corr[season])


        i_arr, j_arr = np.where(~seasonal_obs[1][0, :, :].mask)
        season_to_std_diff[season] = np.ma.masked_all_like(seasonal_obs[1][0, :, :])
        season_to_std_diff[season][i_arr, j_arr] = (seasonal_model[:, i_arr, j_arr].std(axis=0) - seasonal_obs[1][:, i_arr, j_arr].std(axis=0)) / seasonal_obs[1][:, i_arr, j_arr].std(axis=0) * 100



        # area averages
        season_to_area_mean_obs[season] = seasonal_obs[1][:, i_arr, j_arr].mean(axis=1)
        season_to_area_mean_model[season] = seasonal_model[:, i_arr, j_arr].mean(axis=1)


        season_to_area_mean_corr[season] = np.corrcoef(season_to_area_mean_obs[season], season_to_area_mean_model[season])[0, 1]
        season_to_area_mean_std_diff[season] = (season_to_area_mean_model[season].std() - season_to_area_mean_obs[season].std()) / season_to_area_mean_obs[season].std() * 100


        # plt.figure()
        # plt.plot(years, seasonal_obs[1][:, i_arr, j_arr].mean(axis=1), label="obs")
        # plt.plot(years, seasonal_model[:, i_arr, j_arr].mean(axis=1), label="model")
        # plt.legend()
        #
        #
        # plt.figure()
        # im = plt.pcolormesh(season_to_corr[season].T)
        # plt.colorbar(im)
        # plt.show()


    # plotting
    plot_utils.apply_plot_params(font_size=8, width_cm=23, height_cm=15)
    nrows = 3
    gs = gridspec.GridSpec(nrows=nrows, ncols=len(season_to_months) + 1, width_ratios=[1, ] * len(season_to_months) + [0.05, ])

    fig = plt.figure()

    xx, yy = bmp_info.get_proj_xy()


    clevs_corr = np.arange(-1, 1.1, 0.1)
    clevs_std_diff = np.arange(-100, 110, 10)
    cmap = "seismic"
    cmap_std = "seismic"

    cs_corr = None
    cs_std = None

    for col, season in enumerate(season_to_months):

        row = 0
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(season)
        if col == 0:
            ax.set_ylabel("Correlation")
        cs_corr = bmp_info.basemap.contourf(xx, yy, season_to_corr[season], levels=clevs_corr, ax=ax, cmap=cmap)
        bmp_info.basemap.drawcoastlines(ax=ax)



        row += 1
        ax = fig.add_subplot(gs[row, col])
        if col == 0:
            ax.set_ylabel(r"$\left( \sigma_{\rm model} - \sigma_{\rm obs.}\right)/\sigma_{\rm obs.} \cdot 100\%$")

        cs_std = bmp_info.basemap.contourf(xx, yy, season_to_std_diff[season], levels=clevs_std_diff, ax=ax, cmap=cmap_std, extend="both")
        bmp_info.basemap.drawcoastlines(ax=ax)



        # area average
        row += 1
        ax = fig.add_subplot(gs[row, col])
        if col == 0:
            ax.set_ylabel("Area-average")

        ax.plot(season_to_years[season], season_to_area_mean_obs[season], "k", label="Obs.")
        ax.plot(season_to_years[season], season_to_area_mean_model[season], "r", label="Mod.")

        ax.set_title(r"$r=" + "{:.2f}, ".format(season_to_area_mean_corr[season]) + r" \varepsilon_{\sigma} = " + "{:.1f}".format(season_to_area_mean_std_diff[season]) + r" \% $")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70)
        assert isinstance(ax, Axes)

        ax.xaxis.set_minor_locator(MultipleLocator())

        ax.grid()

        if col == 0:
            ax.legend(loc="upper left")


    # add colorbars
    col = len(season_to_months)
    row = 0
    ax = fig.add_subplot(gs[row, col])
    plt.colorbar(cs_corr, cax=ax)

    row += 1
    ax = fig.add_subplot(gs[row, col])
    plt.colorbar(cs_std, cax=ax)


    from crcm5.analyse_hdf import common_plot_params
    fig.tight_layout()
    fig.savefig(os.path.join(img_folder, "{}_{}-{}_corr_std.png".format(vname, sim_config.start_year, sim_config.end_year)), bbox_inches="tight", dpi=common_plot_params.FIG_SAVE_DPI)




if __name__ == '__main__':
    main()
