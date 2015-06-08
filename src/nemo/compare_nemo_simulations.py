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
        start_year=2003, end_year=2009):
    crcm5_model_manager = Crcm5ModelDataManager(samples_folder_path=hl_data_path, all_files_in_samples_folder=True)

    varname = "sohefldo"
    season = "Summer"

    season_to_months = {season: range(6, 9)}
    season_months = list(season_to_months[season])




    # Get Nemo manager here only for coordinates and mask
    nemo_offline_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012",
                                                  suffix="grid_T.nc")

    nemo_crcm5_manager = NemoYearlyFilesManager(
        folder="/home/huziy/skynet3_rech1/one_way_coupled_nemo_outputs_1979_1985",
        suffix="grid_T.nc")

    lon2d, lat2d, bmp = nemo_offline_manager.get_coords_and_basemap()

    # Interpolate


    # get nemo sst
    nemo_offline_sst = nemo_offline_manager.get_seasonal_clim_field(start_year=start_year,
                                                                    end_year=end_year,
                                                                    season_to_months=season_to_months,
                                                                    varname=varname)

    nemo_crcm5_sst = nemo_crcm5_manager.get_seasonal_clim_field(start_year=start_year,
                                                                end_year=end_year,
                                                                season_to_months=season_to_months,
                                                                varname=varname)

    nemo_offline_sst = np.ma.masked_where(~nemo_offline_manager.lake_mask, nemo_offline_sst[season])
    nemo_crcm5_sst = np.ma.masked_where(~nemo_offline_manager.lake_mask, nemo_crcm5_sst[season])

    # plt.figure()
    xx, yy = bmp(lon2d.copy(), lat2d.copy())
    # im = bmp.pcolormesh(xx, yy, model_yearmax_ice_conc)
    # bmp.colorbar(im)

    vmin = min(nemo_offline_sst.min(), nemo_crcm5_sst.min())
    vmax = max(nemo_offline_sst.max(), nemo_crcm5_sst.max())

    # plt.figure()
    # b = Basemap()
    # xx, yy = b(lons_obs, lats_obs)
    # im = b.pcolormesh(xx, yy, obs_yearmax_ice_conc)
    # b.colorbar(im)
    # b.drawcoastlines()

    # Plot as usual: model, obs, model - obs
    img_folder = Path("nemo")
    if not img_folder.is_dir():
        img_folder.mkdir()
    img_file = img_folder.joinpath("{}_{}_nemo-offline_vs_nemo-crcm5_{}-{}.png".format(season.lower(),
                                                                                       varname, start_year, end_year))

    fig = plt.figure()
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    all_axes = []

    cmap = cm.get_cmap("jet", 10)
    locator = MaxNLocator(nbins=10)
    ticks = locator.tick_values(vmin=vmin, vmax=vmax)
    norm = BoundaryNorm(boundaries=ticks, ncolors=len(ticks) - 1)

    # Model, Hostetler
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("NEMO+CRCM5")
    bmp.pcolormesh(xx, yy, nemo_crcm5_sst, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    all_axes.append(ax)

    #
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("NEMO-offline")
    im = bmp.pcolormesh(xx, yy, nemo_offline_sst, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    all_axes.append(ax)


    # Obs: Homa
    # ax = fig.add_subplot(gs[0, 2])
    # ax.set_title("MODIS")
    # im = bmp.pcolormesh(xx, yy, obs_sst_clim, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    # all_axes.append(ax)

    plt.colorbar(im, cax=fig.add_subplot(gs[0, -1]), ticks=ticks)

    for the_ax in all_axes:
        bmp.drawcoastlines(ax=the_ax)

    fig.savefig(str(img_file), bbox_inches="tight")
    plt.close(fig)


def main():
    validate_seas_mean_lswt_from_hostetler_and_nemo_with_homa(start_year=1981, end_year=1985)


if __name__ == '__main__':
    import application_properties

    plot_utils.apply_plot_params(font_size=10, width_cm=25, height_cm=8)
    application_properties.set_current_directory()
    main()
