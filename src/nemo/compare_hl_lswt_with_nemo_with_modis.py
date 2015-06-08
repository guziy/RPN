from netCDF4 import Dataset, num2date
from pathlib import Path
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.spatial import KDTree
from scipy.spatial.ckdtree import cKDTree
from crcm5.model_data import Crcm5ModelDataManager
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from util import plot_utils
from util.geo import lat_lon

__author__ = 'huziy'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(
        hl_data_path="/home/huziy/skynet3_rech1/CRCM_GL_simulation/all_files",
        start_year=2003, end_year=2006):
    crcm5_model_manager = Crcm5ModelDataManager(samples_folder_path=hl_data_path, all_files_in_samples_folder=True)

    varname = "L1"
    season = "Fall"

    season_to_months = {season: range(9, 11)}
    season_months = list(season_to_months[season])

    print(season_months, season_to_months)

    hl_lake_temp_clim = crcm5_model_manager.get_mean_field(start_year=start_year, end_year=end_year, var_name=varname,
                                                           level=1.1, months=season_months)

    hl_lake_temp_clim = hl_lake_temp_clim[13:-13, 13:-13]

    print("hl_lake_temp_clim.shape = ", hl_lake_temp_clim.shape)




    lon_hl, lat_hl = crcm5_model_manager.lons2D, crcm5_model_manager.lats2D



    print("lon_hl.shape = ", lon_hl.shape)

    print(lon_hl.min(), lon_hl.max())
    print(lat_hl.min(), lat_hl.max())

    # Get Nemo manager here only for coordinates and mask
    nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012",
                                          suffix="icemod.nc")

    lon2d, lat2d, bmp = nemo_manager.get_coords_and_basemap()

    # Interpolate hostetler's lake fraction to the model's grid
    hl_lake_temp_clim -= 273.15
    xs, ys, zs = lat_lon.lon_lat_to_cartesian(lon_hl.flatten(), lat_hl.flatten())
    print(xs.shape)
    ktree = cKDTree(data=list(zip(xs, ys, zs)))

    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon2d.flatten(), lat2d.flatten())
    dists, inds = ktree.query(np.asarray(list(zip(xt, yt, zt))))
    print(inds[:20])
    print(type(inds))
    print(inds.min(), inds.max())
    hl_lake_temp_clim = hl_lake_temp_clim.flatten()[inds].reshape(lon2d.shape)





    # get nemo and observed sst
    obs_sst, nemo_sst, _, _, _ = nemo_manager.get_nemo_and_homa_seasonal_mean_sst(start_year=start_year,
                                                                                  end_year=end_year,
                                                                                  season_to_months=season_to_months)

    obs_sst_clim = np.ma.mean([obs_sst[y][season] for y in range(start_year, end_year + 1)], axis=0)
    nemo_sst_clim = np.ma.mean([nemo_sst[y][season] for y in range(start_year, end_year + 1)], axis=0)


    obs_sst_clim = np.ma.masked_where(~nemo_manager.lake_mask, obs_sst_clim)
    nemo_sst_clim = np.ma.masked_where(~nemo_manager.lake_mask, nemo_sst_clim)
    hl_lake_temp_clim = np.ma.masked_where(~nemo_manager.lake_mask, hl_lake_temp_clim)

    # plt.figure()
    xx, yy = bmp(lon2d.copy(), lat2d.copy())
    # im = bmp.pcolormesh(xx, yy, model_yearmax_ice_conc)
    # bmp.colorbar(im)

    vmin = min(obs_sst_clim.min(), nemo_sst_clim.min(), hl_lake_temp_clim.min())
    vmax = max(obs_sst_clim.max(), nemo_sst_clim.max(), hl_lake_temp_clim.max())

    print("vmin={}; vmax={}".format(vmin, vmax))

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
    img_file = img_folder.joinpath("validate_{}_lswt_hostetler_nemo_vs_homa_{}-{}.png".format(
        season, start_year, end_year))

    fig = plt.figure()
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    all_axes = []

    cmap = cm.get_cmap("jet", 10)
    locator = MaxNLocator(nbins=10)
    ticks = locator.tick_values(vmin=vmin, vmax=vmax)
    norm = BoundaryNorm(boundaries=ticks, ncolors=len(ticks) - 1)

    # Model, Hostetler
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Hostetler+CRCM5")
    bmp.pcolormesh(xx, yy, hl_lake_temp_clim, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    all_axes.append(ax)

    #
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("NEMO-offline")
    im = bmp.pcolormesh(xx, yy, nemo_sst_clim, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    all_axes.append(ax)


    # Obs: Homa
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("MODIS")
    im = bmp.pcolormesh(xx, yy, obs_sst_clim, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    all_axes.append(ax)

    plt.colorbar(im, cax=fig.add_subplot(gs[0, -1]), ticks=ticks)

    for the_ax in all_axes:
        bmp.drawcoastlines(ax=the_ax)

    fig.savefig(str(img_file), bbox_inches="tight")
    plt.close(fig)


def main():
    validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(start_year=2003, end_year=2006)


if __name__ == '__main__':
    import application_properties
    plot_utils.apply_plot_params(font_size=10, width_cm=25, height_cm=8)
    application_properties.set_current_directory()
    main()