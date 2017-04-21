from collections import OrderedDict
from pathlib import Path

import xarray as xr
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from scipy.spatial import KDTree

from lake_effect_snow import common_params

import matplotlib.pyplot as plt

from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from util import plot_utils

import numpy as np
import pickle

from util.geo import lat_lon
from math import ceil

def plot_flow_vectors_cartopy(lons, lats, uu, vv, flow_speed, grid_shape=None):

    """
    uu and vv are assumed to be in geographic coordinates 
    :param lons: 
    :param lats: 
    :param uu: 
    :param vv: 
    """



    import cartopy

    fig = plt.figure()
    ax = plt.axes(projection=cartopy.crs.PlateCarree())

    ax.add_feature(cartopy.feature.LAND)
    # ax.add_feature(cartopy.feature.OCEAN)
    # ax.add_feature(cartopy.feature.COASTLINE)
    # ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    # ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    # ax.add_feature(cartopy.feature.RIVERS)

    ax.set_extent([lons[0, 0], lats[0, 0], lons[-1, -1], lats[-1, -1]])


    stride = 3
    ax.quiver(lons[::stride, ::stride], lats[::stride, ::stride], uu[::stride, ::stride], vv[::stride, ::stride])

    fig.savefig("nemo/circ_annual_mean_cartopy.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_flow_vectors_basemap(lons, lats, uu, vv, flow_speed, subregion:list=None, grid_shape=None, ax:Axes=None,
                              streamplot=False, draw_colorbar=True):
    from mpl_toolkits.basemap import Basemap

    if grid_shape is None:
        grid_shape = (300, 300)

    if subregion is None:
        subregion = [0, 1, 0, 1]

    if ax is None:
        fig = plt.figure()



    nx, ny = lons.shape





    b = Basemap(lon_0=180,
                llcrnrlon=lons[35, 35],
                llcrnrlat=lats[35, 35],
                urcrnrlon=lons[nx // 2, ny // 2],
                urcrnrlat=lats[nx // 2, ny // 2],
                resolution="i", area_thresh=2000)

    # im = b.pcolormesh(xx, yy, flow_speed)
    # b.colorbar(im)
    stride = 6



    uu1, vv1 = b.rotate_vector(uu, vv, lons, lats)

    lons_g, lats_g, xx_g, yy_g = b.makegrid(*grid_shape, returnxy=True)

    nx, ny = lons_g.shape
    i_start, i_end = int(nx * subregion[0]), int(nx * subregion[1])
    j_start, j_end = int(ny * subregion[2]), int(ny * subregion[3])



    xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_g.flatten(), lats_g.flatten())
    xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())
    ktree = KDTree(data=list(zip(xs, ys, zs)))

    dists, inds = ktree.query(list(zip(xt, yt, zt)))

    uu_to_plot = uu1.flatten()[inds].reshape(lons_g.shape)
    vv_to_plot = vv1.flatten()[inds].reshape(lons_g.shape)
    flow_speed_to_plot = flow_speed.flatten()[inds].reshape(lons_g.shape)



    clevs = [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16]
    ncolors = len(clevs) - 1
    norm = BoundaryNorm(clevs, ncolors)
    cmap = cm.get_cmap("gist_ncar_r", ncolors)

    im = b.pcolormesh(xx_g, yy_g, flow_speed_to_plot, alpha=0.5, cmap=cmap, norm=norm)
    if not streamplot:

        b.quiver(xx_g[i_start:i_end:stride, j_start:j_end:stride], yy_g[i_start:i_end:stride, j_start:j_end:stride],
                      uu_to_plot[i_start:i_end:stride, j_start:j_end:stride], vv_to_plot[i_start:i_end:stride, j_start:j_end:stride],
                      headlength=2, headaxislength=2, headwidth=4, units="inches", color="k")



    else:
        b.streamplot(xx_g, yy_g, uu_to_plot, vv_to_plot, linewidth=0.4, density=3, arrowstyle="fancy", arrowsize=0.4, ax=ax, color="k")


    # im = b.contourf(xx_g, yy_g, flow_speed_to_plot, levels=clevs, alpha=0.5, cmap=cmap, norm=norm)


    cb = b.colorbar(im, location="bottom")
    cb.ax.set_visible(draw_colorbar)


    b.drawcoastlines(linewidth=0.3, ax=ax)

    if ax is None:
        fig.savefig("nemo/circ_annual_mean_basemap.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    return im


def get_annual_mean_flow(nemo_manager: NemoYearlyFilesManager, start_year:int, end_year:int, level:int=0):
    dirpath = Path(nemo_manager.data_folder)

    cache_file = Path("nemo_annual_wind_tmp_{}-{}_{}.bin".format(start_year, end_year, dirpath.name))

    if cache_file.exists():
        return pickle.load(cache_file.open("rb"))





    ds_u = xr.open_mfdataset(str(dirpath / "*{}*{}*_grid_U.nc".format(start_year, end_year)))
    ds_v = xr.open_mfdataset(str(dirpath / "*{}*{}*_grid_V.nc".format(start_year, end_year)))


    uu = ds_u["vozocrtx"][:, level, :, :].mean(dim="time_counter").values
    vv = ds_v["vomecrty"][:, level, :, :].mean(dim="time_counter").values


    lons, lats = ds_u["nav_lon"][:].values, ds_v["nav_lat"][:].values

    flow_speed = (uu ** 2 + vv ** 2) ** 0.5

    flow_speed = np.ma.masked_where(~nemo_manager.lake_mask, flow_speed)


    print(uu.shape)
    print(flow_speed.shape)

    uu1 = uu / flow_speed
    vv1 = vv / flow_speed

    pickle.dump([lons, lats, uu1, vv1, flow_speed], cache_file.open("wb"))

    return lons, lats, uu1, vv1, flow_speed



def get_seasonal_flows(data_dir="", start_year=1980, end_year=2010, season_to_months=None, level=0):


    """

    :param data_dir: 
    :param start_year: 
    :param end_year: 
    :param season_to_months: 
    :param level: 
    :return: (lons, lats, {season: (uu, vv, flow_speed)}) 
    """

    nemo_manager_U = NemoYearlyFilesManager(data_dir, suffix="grid_U.nc")
    nemo_manager_V = NemoYearlyFilesManager(data_dir, suffix="grid_V.nc")

    u_comp_name = "vozocrtx"
    v_comp_name = "vomecrty"


    dirpath = Path(data_dir)

    cache_file = Path("nemo_seasonal_wind_{}-{}_{}_{}_lev{}.bin".format(
        start_year, end_year, dirpath.name, "_".join([s for s in season_to_months]), level))


    if cache_file.exists():
        return pickle.load(cache_file.open("rb"))


    uu = nemo_manager_U.get_seasonal_clim_field(start_year=start_year, end_year=end_year, season_to_months=season_to_months, varname=u_comp_name, level_index=level)
    vv = nemo_manager_V.get_seasonal_clim_field(start_year=start_year, end_year=end_year, season_to_months=season_to_months, varname=v_comp_name, level_index=level)


    # Cache to reuse in future
    pickle.dump([nemo_manager_U.lons, nemo_manager_U.lats, {season: (uu[season], vv[season]) for season in season_to_months}], cache_file.open("wb"))

    return nemo_manager_U.lons, nemo_manager_U.lats, {season: (uu[season], vv[season]) for season in season_to_months}




def plot_seasonal_circulation_as_subplots(start_year=1995, end_year=2010,
                                          data_dir="/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/zdf_gls_dt_and_sbc_30min",
                                          season_to_months=None, level_index=0):




    # get the data
    lons, lats, season_to_fields = get_seasonal_flows(data_dir=data_dir, start_year=start_year, end_year=end_year, season_to_months=season_to_months, level=level_index)



    img_folder = Path("nemo/circulation_plots/" + Path(data_dir).name)

    if not img_folder.exists():
        img_folder.mkdir(parents=True)

    img_name = "circ_{}-{}_lev{}.png".format(start_year, end_year, level_index)


    nrows = 3
    nsubplots = len(season_to_months)
    ncols = ceil(nsubplots / (nrows - 1))
    gs = GridSpec(nrows, ncols, wspace=0.01, hspace=0, height_ratios=[1,] * (nrows - 1) + [0.05, ])

    plot_utils.apply_plot_params(font_size=8, width_cm=8 * ncols, height_cm=min(4.5 * (nrows - 1), 25))

    fig = plt.figure()

    plot_ind = 0
    for season in season_to_months:

        row = plot_ind // ncols
        col = plot_ind % ncols

        uu, vv = season_to_fields[season]
        flow_speed = (uu ** 2 + vv ** 2) ** 0.5

        uu1, vv1 = uu / flow_speed, vv / flow_speed

        ax = fig.add_subplot(gs[row, col])

        assert isinstance(ax, Axes)
        ax.set_frame_on(False)

        ax.text(0.01, 0.1, season, va="bottom", ha="left", fontsize=12, transform=ax.transAxes)
        im = plot_flow_vectors_basemap(lons=lons, lats=lats, uu=uu1, vv=vv1, flow_speed=flow_speed, ax=ax,
                                       draw_colorbar=(col == 0) and (row == nrows - 2),
                                       streamplot=False)

        plot_ind += 1


    # plt.colorbar(im, cax=fig.add_subplot(gs[nrows - 1, 0]), orientation="horizontal")

    img_path = img_folder / img_name
    fig.savefig(str(img_path), bbox_inches="tight", dpi=300)




def main_bak():
    dirpath = "/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/zdf_gls_dt_and_sbc_30min"
    dirpath = Path(dirpath)


    nemo_manager = NemoYearlyFilesManager(folder=str(dirpath))

    plot_utils.apply_plot_params(width_cm=20, font_size=10)

    lons, lats, uu1, vv1, flow_speed = get_annual_mean_flow(nemo_manager, start_year=1998, end_year=1998, level=0)

    lons[lons < 0] += 360
    plot_flow_vectors_basemap(lons, lats, uu1, vv1, flow_speed)

    lons[lons > 180] -= 360
    # plot_flow_vectors_cartopy(lons, lats, uu1, vv1, flow_speed)


def main():
    season_to_months = OrderedDict([
        ("Winter", [12, 1, 2]), ("Spring", [3, 4, 5]), ("Summer", [6, 7, 8]), ("Fall", [9, 10, 11])
    ])


    # plot_seasonal_circulation_as_subplots(start_year=1999, end_year=2010,
    #                                       data_dir="/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3/EXP_GLK_LIM3_1980/zdf_gls_dt_and_sbc_30min",
    #                                       level_index=0, season_to_months=season_to_months
    #                                       )

    # plot_seasonal_circulation_as_subplots(start_year=1980, end_year=2010,
    #                                       data_dir="/HOME/huziy/skynet3_rech1/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_CC_drivenby_CRCM5_CanESM2_RCP85/EXP00/cc_canesm2_outputs",
    #                                       level_index=0, season_to_months=season_to_months
    #                                       )

    plot_seasonal_circulation_as_subplots(start_year=1980, end_year=2010,
                                          data_dir="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO",
                                          level_index=0, season_to_months=season_to_months
                                          )



if __name__ == '__main__':
    main()