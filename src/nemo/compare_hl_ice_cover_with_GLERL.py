from netCDF4 import Dataset, num2date
from pathlib import Path
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from scipy.spatial.ckdtree import cKDTree
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from util.geo import lat_lon

__author__ = 'huziy'

from crcm5.model_data import Crcm5ModelDataManager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def validate_yearmax_ice_cover_from_hostetler_with_glerl(path="/home/huziy/skynet3_rech1/CRCM_GL_simulation/all_files",
                                                         start_year=2003, end_year=2009):
    model_manager = Crcm5ModelDataManager(samples_folder_path=path, all_files_in_samples_folder=True)

    varname = "LC"

    hl_icecov_yearly_max = model_manager.get_yearmax_fields(start_year=start_year, end_year=end_year, var_name=varname)

    hl_icecov_yearly_max_clim = np.mean(list(hl_icecov_yearly_max.values()), axis=0)




    # Get Nemo manager here only for coordinates and mask
    nemo_manager = NemoYearlyFilesManager(folder="/home/huziy/skynet3_rech1/offline_glk_output_daily_1979-2012",
                                          suffix="icemod.nc")

    lon2d, lat2d, bmp = nemo_manager.get_coords_and_basemap()

    # Interpolate hostetler's lake fraction to the model's grid
    hl_icecov_yearly_max_clim = model_manager.interpolate_data_to(hl_icecov_yearly_max_clim, lon2d, lat2d)
    hl_icecov_yearly_max_clim = np.ma.masked_where(~nemo_manager.lake_mask, hl_icecov_yearly_max_clim)

    model_lake_avg_ts = []
    for the_year in range(start_year, end_year + 1):
        model_lake_avg_ts.append(hl_icecov_yearly_max[the_year][nemo_manager.lake_mask].mean())



    # plt.figure()
    xx, yy = bmp(lon2d.copy(), lat2d.copy())
    # im = bmp.pcolormesh(xx, yy, model_yearmax_ice_conc)
    # bmp.colorbar(im)

    # Read and interpolate obs
    path_to_obs = "/RESCUE/skynet3_rech1/huziy/nemo_obs_for_validation/glerl_icecov.nc"

    obs_varname = "ice_cover"
    obs_lake_avg_ts = []
    with Dataset(path_to_obs) as ds:
        time_var = ds.variables["time"]

        lons_obs = ds.variables["lon"][:]
        lats_obs = ds.variables["lat"][:]

        dates = num2date(time_var[:], time_var.units)
        nx, ny = lons_obs.shape

        data = ds.variables[obs_varname][:]
        data = np.ma.masked_where((data > 100) | (data < 0), data)
        print(data.min(), data.max())
        panel = pd.Panel(data=data, items=dates, major_axis=range(nx), minor_axis=range(ny))

        panel = panel.select(lambda d: start_year <= d.year <= end_year)
        the_max_list = []
        for key, g in panel.groupby(lambda d: d.year, axis="items"):
            the_max_field = np.ma.max(np.ma.masked_where((g.values > 100) | (g.values < 0), g.values), axis=0)
            obs_lake_avg_ts.append(the_max_field.mean())
            the_max_list.append(the_max_field)

        obs_yearmax_ice_conc = np.ma.mean(the_max_list, axis=0) / 100.0

        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_obs.flatten(), lats_obs.flatten())
        ktree = cKDTree(list(zip(xs, ys, zs)))

        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lon2d.flatten(), lat2d.flatten())
        dists, inds = ktree.query(list(zip(xt, yt, zt)))

        obs_yearmax_ice_conc_interp = obs_yearmax_ice_conc.flatten()[inds].reshape(lon2d.shape)
        obs_yearmax_ice_conc_interp = np.ma.masked_where(~nemo_manager.lake_mask, obs_yearmax_ice_conc_interp)


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
    img_file = img_folder.joinpath("validate_yearmax_icecov_hl_vs_glerl_{}-{}.pdf".format(start_year, end_year))

    fig = plt.figure()
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.05])
    all_axes = []

    cmap = cm.get_cmap("jet", 10)
    diff_cmap = cm.get_cmap("RdBu_r", 10)

    # Model
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Hostetler")
    bmp.pcolormesh(xx, yy, hl_icecov_yearly_max_clim, cmap=cmap, vmin=0, vmax=1)
    all_axes.append(ax)



    # Obs
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("GLERL")
    im = bmp.pcolormesh(xx, yy, obs_yearmax_ice_conc_interp, cmap=cmap, vmin=0, vmax=1)
    all_axes.append(ax)

    plt.colorbar(im, cax=fig.add_subplot(gs[0, -1]))



    # Biases
    ax = fig.add_subplot(gs[1, :])
    ax.set_title("Hostetler - GLERL")
    im = bmp.pcolormesh(xx, yy, hl_icecov_yearly_max_clim - obs_yearmax_ice_conc_interp,
                        cmap=diff_cmap, vmin=-1, vmax=1)
    bmp.colorbar(im, ax=ax)
    all_axes.append(ax)

    for the_ax in all_axes:
        bmp.drawcoastlines(ax=the_ax)

    fig.savefig(str(img_file), bbox_inches="tight")
    plt.close(fig)


    # Plot lake aversged ice concentrations
    fig = plt.figure()
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.plot(range(start_year, end_year + 1), model_lake_avg_ts, "b", lw=2, label="Hostetler")
    plt.plot(range(start_year, end_year + 1), np.asarray(obs_lake_avg_ts) / 100.0, "r", lw=2, label="GLERL")

    plt.grid()
    plt.legend(loc=3)
    fig.savefig(str(img_folder.joinpath("lake_avg_iceconc_hostetler_offline_vs_GLERL.pdf")), bbox_inches="tight")


def main():
    pass


if __name__ == '__main__':
    import application_properties

    application_properties.set_current_directory()
    validate_yearmax_ice_cover_from_hostetler_with_glerl()