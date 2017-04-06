from netCDF4 import Dataset, num2date
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.spatial import KDTree
from scipy.spatial.ckdtree import cKDTree
from crcm5.model_data import Crcm5ModelDataManager
from data.oisst import OISSTManager
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(
        hl_data_path="/home/huziy/skynet3_rech1/CRCM_GL_simulation/all_files",
        start_year=2003, end_year=2003):
    """
    Note: degrees plotted are in C
    :param hl_data_path:
    :param start_year:
    :param end_year:
    """
    crcm5_model_manager = Crcm5ModelDataManager(samples_folder_path=hl_data_path, all_files_in_samples_folder=True)

    varname = "L1"
    season = "September"

    use_noaa_oisst = True
    obs_label = "NOAA OISST" if use_noaa_oisst else "MODIS"

    clevs = np.arange(-21, 22, 2)
    norm = BoundaryNorm(clevs, len(clevs) - 1)
    cmap = cm.get_cmap("viridis", len(clevs) - 1)


    clevs_bias = np.arange(-7.5, 7.6, 1)
    norm_bias = BoundaryNorm(clevs_bias, len(clevs_bias) - 1)
    cmap_bias = cm.get_cmap("bwr", len(clevs_bias) - 1)



    season_to_months = {"Fall": range(9, 11),
                        "Summer": range(6, 9),
                        "Winter": [1, 2, 12],
                        "June": [6, ],
                        "July": [7, ],
                        "August": [8, ],
                        "September": [9, ],
                        "October": [10,],
                        "November": [11, ],
                        "December": [12, ],
                        "January": [1, ],
                        "February": [2],
                        "March": [3]
                        }


    # hl_lake_temp_clim = crcm5_model_manager.get_mean_field(start_year=start_year, end_year=end_year, var_name=varname,
    #                                                        level=1.1, months=season_months)


    lon_hl, lat_hl = crcm5_model_manager.lons2D, crcm5_model_manager.lats2D



    print("lon_hl.shape = ", lon_hl.shape)

    print(lon_hl.min(), lon_hl.max())
    print(lat_hl.min(), lat_hl.max())

    # Get Nemo manager here only for coordinates and mask
    # nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012",
    #                                       suffix="icemod.nc")

    # nemo_manager = NemoYearlyFilesManager(folder="/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/zdf_gls_dt_and_sbc_5min",
    #                                       suffix="grid_T.nc")


    model_label = "CRCM5_NEMO"

    nemo_manager = NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO",
                                          suffix="grid_T.nc")

    lon2d, lat2d, bmp = nemo_manager.get_coords_and_basemap(resolution="l")




    # plt.figure()
    # im = plt.pcolormesh(hl_lake_temp_clim.T)
    # plt.colorbar(im)
    # plt.show()
    #
    # if True:
    #     raise Exception()


    xs, ys, zs = lat_lon.lon_lat_to_cartesian(lon_hl.flatten(), lat_hl.flatten())
    print(xs.shape)
    ktree = cKDTree(data=list(zip(xs, ys, zs)))

    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon2d.flatten(), lat2d.flatten())
    dists, inds = ktree.query(np.asarray(list(zip(xt, yt, zt))))





    # get nemo and observed sst
    nemo_sst, obs_sst, _, _, _ = nemo_manager.get_nemo_and_homa_seasonal_mean_sst(start_year=start_year,
                                                                                  end_year=end_year,
                                                                                  season_to_months={season: season_to_months[season]},
                                                                                  use_noaa_oisst=use_noaa_oisst)




    if use_noaa_oisst:
        manager = OISSTManager(thredds_baseurl="/BIG1/huziy/noaa_oisst_daily")
        obs_sst_clim = manager.get_seasonal_clim_interpolate_to(lons=lon2d, lats=lat2d,
                                                           start_year=start_year, end_year=end_year,
                                                           season_to_months=season_to_months, vname="sst")

        obs_sst_clim = obs_sst_clim[season]
        obs_sst_clim = np.ma.masked_where(np.isnan(obs_sst_clim), obs_sst_clim)

    else:
        # Convert to Celsius
        obs_sst_clim = np.ma.mean([obs_sst[y][season] for y in range(start_year, end_year + 1)], axis=0) - 273.15




    obs_sst_clim = np.ma.masked_where(~nemo_manager.lake_mask, obs_sst_clim)
    nemo_sst_clim = np.ma.mean([nemo_sst[y][season] for y in range(start_year, end_year + 1)], axis=0)

    nemo_sst_clim = np.ma.masked_where(~nemo_manager.lake_mask, nemo_sst_clim)






    # plt.figure()
    xx, yy = bmp(lon2d, lat2d)
    # im = bmp.pcolormesh(xx, yy, model_yearmax_ice_conc)
    # bmp.colorbar(im)






    # plt.figure()
    # b = Basemap()
    # xx, yy = b(lons_obs, lats_obs)
    # im = b.pcolormesh(xx, yy, obs_yearmax_ice_conc)
    # b.colorbar(im)
    # b.drawcoastlines()

    # Plot as usual: model, obs, model - obs
    img_folder = Path("nemo/hostetler")
    if not img_folder.is_dir():
        img_folder.mkdir()
    img_file = img_folder.joinpath("validate_{}_lswt_hostetler_nemolatest_vs_{}_{}-{}.png".format(
        season, obs_label, start_year, end_year))

    fig = plt.figure()
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.05])
    all_axes = []


    col = 0
    # Model, Hostetler
    # ax = fig.add_subplot(gs[0, 0])
    # ax.set_title("Hostetler+CRCM5")
    # bmp.pcolormesh(xx, yy, hl_lake_temp_clim, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    # col += 1
    # all_axes.append(ax)

    #
    ax = fig.add_subplot(gs[0, col])
    ax.set_title(model_label)
    im = bmp.pcolormesh(xx, yy, nemo_sst_clim, cmap=cmap, norm=norm)
    all_axes.append(ax)
    col += 1


    # Obs: MODIS or NOAA OISST
    ax = fig.add_subplot(gs[0, col])
    ax.set_title(obs_label)
    im = bmp.pcolormesh(xx, yy, obs_sst_clim, cmap=cmap, norm=norm)
    all_axes.append(ax)
    col += 1

    plt.colorbar(im, cax=fig.add_subplot(gs[1, :col]), extend="both", orientation="horizontal")



    ax = fig.add_subplot(gs[0, col])
    ax.set_title("{} - {}".format(model_label, obs_label))
    im = bmp.pcolormesh(xx, yy, nemo_sst_clim - obs_sst_clim, cmap=cmap_bias, norm=norm_bias)
    plt.colorbar(im, orientation="horizontal", cax=fig.add_subplot(gs[1, col]), ticks=[c for c in clevs_bias if c % 1 == 0])
    all_axes.append(ax)
    col += 1

    for the_ax in all_axes:
        bmp.drawcoastlines(ax=the_ax, linewidth=0.3)

    print("Saving {}".format(img_file))
    fig.savefig(str(img_file), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(start_year=2003, end_year=2006)


if __name__ == '__main__':
    import application_properties
    plot_utils.apply_plot_params(font_size=10, width_cm=25, height_cm=8)
    application_properties.set_current_directory()
    main()