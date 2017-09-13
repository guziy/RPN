from collections import OrderedDict
from pathlib import Path

from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec

from crcm5.analyse_hdf.run_config import RunConfig
from data.oisst import OISSTManager
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'

import matplotlib.pyplot as plt
import numpy as np


def print_arr_limits(arr, label):
    print("{}: {} ... {}".format(label, arr.min(), arr.max()))



def get_seasonal_sst_from_crcm5_outputs(sim_label, start_year=1980, end_year=2010, season_to_months=None,
                                        lons_target=None, lats_target=None):




    from lake_effect_snow.default_varname_mappings import T_AIR_2M
    from lake_effect_snow.default_varname_mappings import U_WE
    from lake_effect_snow.default_varname_mappings import V_SN
    from lake_effect_snow.base_utils import VerticalLevel
    from rpn import level_kinds
    from lake_effect_snow import default_varname_mappings
    from data.robust import data_source_types

    from data.robust.data_manager import DataManager


    sim_configs = {

        sim_label: RunConfig(data_path="/RECH2/huziy/coupling/GL_440x260_0.1deg_GL_with_Hostetler/Samples_selected",
                             start_year=start_year, end_year=end_year, label=sim_label),

    }

    r_config = sim_configs[sim_label]

    vname_to_level = {
        T_AIR_2M: VerticalLevel(1, level_kinds.HYBRID),
        U_WE: VerticalLevel(1, level_kinds.HYBRID),
        V_SN: VerticalLevel(1, level_kinds.HYBRID),
        default_varname_mappings.LAKE_WATER_TEMP: VerticalLevel(1, level_kinds.ARBITRARY)
    }




    vname_map = {}

    vname_map.update(default_varname_mappings.vname_map_CRCM5)



    store_config = {
        "base_folder": r_config.data_path,
        "data_source_type": data_source_types.SAMPLES_FOLDER_FROM_CRCM_OUTPUT_VNAME_IN_FNAME,
        "varname_mapping": vname_map,
        "level_mapping": vname_to_level,
        "offset_mapping": default_varname_mappings.vname_to_offset_CRCM5,
        "multiplier_mapping": default_varname_mappings.vname_to_multiplier_CRCM5,
    }


    dm = DataManager(store_config=store_config)


    season_to_year_to_mean = dm.get_seasonal_means(start_year=start_year, end_year=end_year,
                                                   season_to_months=season_to_months,
                                                   varname_internal=default_varname_mappings.LAKE_WATER_TEMP)

    result = {}

    # fill in the result dictionary with seasonal means
    for season in season_to_months:
        result[season] = np.array([field for field in season_to_year_to_mean[season].values()]).mean(axis=0)



    # interpolate the data
    if lons_target is not None:
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_target.flatten(), lats_target.flatten())

        dists, inds = dm.get_kdtree().query(list(zip(xt, yt, zt)))
        for season in season_to_months:
            result[season] = result[season].flatten()[inds].reshape(lons_target.shape)

    return result




def validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(
        hl_data_path="/home/huziy/skynet3_rech1/CRCM_GL_simulation/all_files",
        start_year=2003, end_year=2003, season_to_months=None):
    """
    Note: degrees plotted are in C
    :param hl_data_path:
    :param start_year:
    :param end_year:
    """


    # crcm5_model_manager = Crcm5ModelDataManager(samples_folder_path=hl_data_path, all_files_in_samples_folder=True)

    model_label1 = "CRCM5_HL"
    plot_hl_biases = True



    use_noaa_oisst = True
    obs_label = "NOAA OISST" if use_noaa_oisst else "MODIS"

    clevs = np.arange(0, 22, 1)
    norm = BoundaryNorm(clevs, len(clevs) - 1)
    cmap = cm.get_cmap("viridis", len(clevs) - 1)


    clevs_bias = np.arange(-5.5, 6.5, 1)
    norm_bias = BoundaryNorm(clevs_bias, len(clevs_bias) - 1)
    cmap_bias = cm.get_cmap("bwr", len(clevs_bias) - 1)




    if season_to_months is None:
        season_to_months = OrderedDict([
            ("Winter", [1, 2, 12]),
            ("Spring", [3, 4, 5]),
            ("Summer", range(6, 9)),
            ("Fall", range(9, 11)),
        ])



    # hl_lake_temp_clim = crcm5_model_manager.get_mean_field(start_year=start_year, end_year=end_year, var_name=varname,
    #                                                        level=1.1, months=season_months)




    # Get Nemo manager here only for coordinates and mask
    # nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012",
    #                                       suffix="icemod.nc")

    # nemo_manager = NemoYearlyFilesManager(folder="/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/zdf_gls_dt_and_sbc_5min",
    #                                       suffix="grid_T.nc")

    model_label2 = "CRCM5_NEMO"
    nemo_manager = NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO",
                                          suffix="grid_T.nc")


    # model_label2 = "NEMO_offline_CRCM5_CanESM2"
    # nemo_manager = NemoYearlyFilesManager(folder="/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_CC_drivenby_CRCM5_CanESM2_RCP85/EXP00/cc_canesm2_outputs",
    #                                       suffix="grid_T.nc")

    # model_label2 = "NEMO_offline"

    # nemo_manager = NemoYearlyFilesManager(folder="/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/zdf_gls_dt_and_sbc_30min",
    #                                       suffix="grid_T.nc")

    img_folder = Path("nemo/{}".format(model_label2))



    #lon2d, lat2d, bmp = nemo_manager.get_coords_and_basemap(resolution="l", subregion=[0.06, 0.5, 0.06, 0.5])
    lon2d, lat2d, bmp = nemo_manager.get_coords_and_basemap(resolution="l", subregion=[0.1, 0.5, 0.15, 0.5], area_thresh=2000)

    xx, yy = bmp(lon2d, lat2d)


    # get nemo and observed sst
    nemo_sst, obs_sst, _, _, _ = nemo_manager.get_nemo_and_homa_seasonal_mean_sst(start_year=start_year,
                                                                                  end_year=end_year,
                                                                                  season_to_months=season_to_months,
                                                                                  use_noaa_oisst=use_noaa_oisst)


    obs_sst_clim = {}



    if use_noaa_oisst:
        manager = OISSTManager(thredds_baseurl="/BIG1/huziy/noaa_oisst_daily")
        obs_sst_clim = manager.get_seasonal_clim_interpolate_to(lons=lon2d, lats=lat2d,
                                                                start_year=start_year, end_year=end_year,
                                                                season_to_months=season_to_months, vname="sst")

        for season in season_to_months:
            obs_sst_clim[season] = np.ma.masked_where(np.isnan(obs_sst_clim[season]), obs_sst_clim[season])

    else:
        # Convert to Celsius
        for season in season_to_months:
            obs_sst_clim[season] = np.ma.mean([obs_sst[y][season] for y in range(start_year, end_year + 1)], axis=0) - 273.15


    obs_sst_clim = {season: np.ma.masked_where(~nemo_manager.lake_mask, obs_sst_clim[season]) for season in obs_sst_clim}
    nemo_sst_clim = {season: np.ma.mean([nemo_sst[y][season] for y in range(start_year, end_year + 1)], axis=0) for season in season_to_months}
    nemo_sst_clim = {season: np.ma.masked_where(~nemo_manager.lake_mask, nemo_sst_clim[season]) for season in season_to_months}




    hl_sst_clim = {}
    if plot_hl_biases:
        hl_sst_clim = get_seasonal_sst_from_crcm5_outputs(model_label1, start_year=start_year, end_year=end_year,
                                                          season_to_months=season_to_months,
                                                          lons_target=lon2d, lats_target=lat2d)

        hl_sst_clim = {season: np.ma.masked_where(~nemo_manager.lake_mask, hl_sst_clim[season]) for season in season_to_months}

        # Convert to C
        hl_sst_clim = {season: hl_sst_clim[season] - 273.15 for season in season_to_months}






    # plt.figure()

    # im = bmp.pcolormesh(xx, yy, model_yearmax_ice_conc)
    # bmp.colorbar(im)



    # plt.figure()
    # b = Basemap()
    # xx, yy = b(lons_obs, lats_obs)
    # im = b.pcolormesh(xx, yy, obs_yearmax_ice_conc)
    # b.colorbar(im)
    # b.drawcoastlines()

    # Plot as usual: model, obs, model - obs

    if not img_folder.is_dir():
        img_folder.mkdir()

    img_file = img_folder.joinpath("validate_{}_lswt_{}_vs_{}_{}-{}.png".format(
        "_".join([season for season in season_to_months]),
        "_".join([model_label1 if plot_hl_biases else "", model_label2]),
        obs_label,
        start_year, end_year))




    nrows = 3
    plot_utils.apply_plot_params(font_size=8, width_cm=8 * len(season_to_months), height_cm=4.5 * nrows)


    fig = plt.figure()



    gs = GridSpec(nrows=nrows, ncols=len(season_to_months), hspace=0.15, wspace=0.000)
    all_axes = []



    # Model, Hostetler
    # ax = fig.add_subplot(gs[0, 0])
    # ax.set_title("Hostetler+CRCM5")
    # bmp.pcolormesh(xx, yy, hl_lake_temp_clim, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    # col += 1
    # all_axes.append(ax)



    for col, season in enumerate(season_to_months):

        row = 0

        # Obs: MODIS or NOAA OISST
        ax = fig.add_subplot(gs[row, col])
        im_obs = bmp.pcolormesh(xx, yy, obs_sst_clim[season], cmap=cmap, norm=norm)
        # obs values
        cb = bmp.colorbar(im_obs, location="bottom")
        cb.ax.set_visible(col == 0)



        ax.text(0.99, 0.99, season, va="top", ha="right", fontsize=16, transform=ax.transAxes)
        all_axes.append(ax)
        if col == 0:
            ax.set_ylabel(obs_label)

        row += 1


        if plot_hl_biases:
            #plot CRCM5_HL biases (for poster)
            ax = fig.add_subplot(gs[row, col])

            if col == 0:
                ax.set_ylabel("{}\n-\n{}".format(model_label1, obs_label))

            im_bias = bmp.pcolormesh(xx, yy, hl_sst_clim[season] - obs_sst_clim[season], cmap=cmap_bias, norm=norm_bias, ax=ax)
            cb = bmp.colorbar(im_bias, location="bottom", ax=ax)
            cb.ax.set_visible(False)

            all_axes.append(ax)
            row += 1


        ax = fig.add_subplot(gs[row, col])
        if col == 0:
            ax.set_ylabel("{}\n-\n{}".format(model_label2, obs_label))
        im_bias = bmp.pcolormesh(xx, yy, nemo_sst_clim[season] - obs_sst_clim[season], cmap=cmap_bias, norm=norm_bias)
        # common for all bias values
        cb = bmp.colorbar(im_bias, location="bottom", ax=ax)
        cb.ax.set_visible(col == 0)


        all_axes.append(ax)
        row += 1


        print_arr_limits(nemo_sst_clim[season], "NEMO_sst, for {}".format(season))
        print_arr_limits(obs_sst_clim[season], "Obs_sst, for {}".format(season))


    for the_ax in all_axes:
        bmp.drawcoastlines(ax=the_ax, linewidth=0.3)
        the_ax.set_frame_on(False)

    print("Saving {}".format(img_file))
    fig.savefig(str(img_file), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():

    # for m in range(1, 13):
    #     season = calendar.month_name[m]
    #     validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(start_year=2003, end_year=2010, season=season)

    # for season in ["Winter", "Spring", "Summer", "Fall"]:
    #     validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(start_year=2003, end_year=2010, season=season)


    season_to_months = OrderedDict([
            ("Winter", [1, 2, 12]),
            ("Spring", [3, 4, 5]),
            ("Summer", range(6, 9)),
            ("Fall", range(9, 11)),
    ])

    validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(
        start_year=2003, end_year=2010, season_to_months=season_to_months
    )



if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    main()