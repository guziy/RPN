import calendar
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import functools
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.basemap import Basemap
from scipy.spatial import KDTree

from application_properties import main_decorator
from nemo.nemo_yearly_files_manager import NemoYearlyFilesManager
from nemo.nic_cis_ice_cover_manager import CisNicIceManager

import numpy as np
import calendar
import matplotlib.pyplot as plt


# color levels for ice fraction
from util import plot_utils
from util.geo import lat_lon




# =========== Common params for the script
from util.geo.mask_from_shp import get_mask

clevs_ice = np.arange(0, 1.1, 0.1)
bnorm_ice = BoundaryNorm(clevs_ice, len(clevs_ice) - 1)
cmap_ice = cm.get_cmap("viridis", len(clevs_ice) - 1)

# color levels for ice fraction change
clevs_ice_diff = np.arange(-0.55, 0.6, 0.1)
bnorm_ice_diff = BoundaryNorm(clevs_ice_diff, len(clevs_ice_diff))
cmap_ice_diff = cm.get_cmap("bwr", len(clevs_ice_diff) - 1)

img_folder = Path("nemo/ice_validation_plots")



def validate_2d_maps(nemo_managers, obs_manager:CisNicIceManager, start_year=-np.Inf, end_year=np.Inf,
                     season_to_months=None, nemo_icecover_name="soicecov", nemo_field_level_index=0, basemap=None):




    """
    All the compared model fields are interpolated to the obs grid
    :param nemo_managers:
    :param obs_manager:
    :param start_year:
    :param end_year:
    :param season_to_months:
    :param nemo_icecover_name:
    :param nemo_field_level_index:
    :param basemap:
    """



    # read obs data
    obs_data = obs_manager.get_seasonal_mean_climatologies(start_year=start_year, end_year=end_year,
                                                       season_to_months=season_to_months)

    #
    xo, yo, zo = lat_lon.lon_lat_to_cartesian(obs_manager.lons.flatten(), obs_manager.lats.flatten())
    ktree = KDTree(list(zip(xo, yo, zo)))


    label_to_nemo_data = OrderedDict()


    season_to_selected_dates = {season: obs_data[season][-1] for season in obs_data}

    # update the season to months map to exclude seasons that do not have obs data
    season_to_months_ordered = OrderedDict([(s, m) for s, m in season_to_months.items() if s in obs_data])


    print(season_to_selected_dates.keys())


    for label, nemo_manager in nemo_managers.items():
        assert isinstance(nemo_manager, NemoYearlyFilesManager)

        label_to_nemo_data[label] = nemo_manager.get_seasonal_clim_field_for_dates(start_year=start_year, end_year=end_year, varname=nemo_icecover_name,
                                                       season_to_selected_dates=season_to_selected_dates, level_index=nemo_field_level_index,
                                                                                   season_to_months=season_to_months_ordered)


        xm, ym, zm = lat_lon.lon_lat_to_cartesian(nemo_manager.lons.flatten(), nemo_manager.lats.flatten())

        dists, inds = ktree.query(list(zip(xm, ym, zm)))

        # interpolate mean and std fields
        for season in label_to_nemo_data[label]:
            interp_mean_and_std = [label_to_nemo_data[label][season][findex].flatten()[inds].reshape(obs_manager.lons.shape) for findex in [0, 1]]
            label_to_nemo_data[label][season] = tuple(interp_mean_and_std) + label_to_nemo_data[label][season][2:]


    # figure
    #   cols = seasons
    #   rows = (obs, mod1 - obs, mod2 - obs, ..., modn - obs)


    plot_utils.apply_plot_params(width_cm=8 * len(season_to_months), height_cm=5 * (len(nemo_managers) + 1), font_size=8)
    fig = plt.figure()


    xx, yy = basemap(obs_manager.lons, obs_manager.lats)
    gs = GridSpec(nrows=1 + len(label_to_nemo_data), ncols=len(season_to_months_ordered), wspace=0., hspace=0.1)


    # plot obs data
    for col, (season, months) in enumerate(season_to_months_ordered.items()):

        ax = fig.add_subplot(gs[0, col])
        im = basemap.pcolormesh(xx, yy, obs_data[season][0], norm=bnorm_ice, cmap=cmap_ice)
        basemap.drawcoastlines(linewidth=0.3)
        cb = basemap.colorbar(im, location="bottom")


        ax.set_title(season)
        cb.ax.set_visible(col == 0)
        ax.set_frame_on(False)

        if col == 0:
            ax.set_ylabel("Obs.")


    # plot model biases
    for row, (label, nemo_data) in enumerate(label_to_nemo_data.items(), start=1):
        for col, (season, months) in enumerate(season_to_months_ordered.items()):

            ax = fig.add_subplot(gs[row, col])
            im = basemap.pcolormesh(xx, yy, nemo_data[season][0][:] - obs_data[season][0][:], norm=bnorm_ice_diff, cmap=cmap_ice_diff)
            basemap.drawcoastlines(linewidth=0.3)
            cb = basemap.colorbar(im, location="bottom")

            cb.ax.set_visible(col == 0 and row == len(label_to_nemo_data))

            if col == 0:
                ax.set_ylabel(label)

            ax.set_frame_on(False)


    # Save the plot to file
    if not img_folder.exists():
        img_folder.mkdir(parents=True)

    nemo_labels = "_".join(label_to_nemo_data)
    seasons = "_".join(season_to_months_ordered)
    img_file = img_folder.joinpath("{}_{}_{}-{}.png".format(nemo_labels, seasons, start_year, end_year))
    fig.savefig(str(img_file), bbox_inches="tight", dpi=300)






def __map_date_to_seasonyear(d, s_month_start, s_month_count):

    for month in range(s_month_start, s_month_start + s_month_count):
        if month == d.month:
            return d.year

        if month == d.month + 12:
            return d.year - 1

    return -1



def validate_areaavg_annual_max(nemo_configs:dict, obs_manager:CisNicIceManager, start_year=-np.Inf, end_year=np.Inf,
                                season_month_start=11, season_month_count=5, mask_shape_file=""):
    """
    the year of the start of the season corresonds to the aggregated value for the season, i.e. if season starts in Oct 2009 and ends in March 2010, then the maximum value 
    for the season would correspond to 2009
    :param nemo_configs: 
    :param obs_manager: 
    :param start_year: 
    :param end_year: 
    """

    lake_mask_obs = get_mask(obs_manager.lons, obs_manager.lats, shp_path=mask_shape_file) > 0.5

    icefr_obs = obs_manager.get_area_avg_ts(lake_mask_obs, start_year=start_year, end_year=end_year)


    plot_utils.apply_plot_params(width_cm=8, height_cm=5, font_size=8)




    fig = plt.figure()



    ax = icefr_obs.groupby(lambda d: __map_date_to_seasonyear(d, season_month_start, season_month_count)).max().drop(-1).plot(label="Obs.")

    label_to_nemo_ts = OrderedDict()
    for label, nemo_config in nemo_configs.items():
        label_to_nemo_ts[label] = nemo_config.get_area_avg_ts(lake_mask_obs, start_year=start_year, end_year=end_year)
        label_to_nemo_ts[label].groupby(lambda d: __map_date_to_seasonyear(d, season_month_start, season_month_count)).max().drop(-1).plot(ax=ax, label=label)

    ax.legend()
    img_file = img_folder.joinpath("icefr_area_avg_max_{}-{}.png")

    fig.savefig(str(img_file), bbox_inches="tight", dpi=300)




@main_decorator
def main():

    obs_data_path = "/RESCUE/skynet3_rech1/huziy/obs_data_for_HLES/interploated_to_the_same_grid/GL_0.1_452x260/cis_nic_glerl_interpolated_lc.nc"


    obs_manager = CisNicIceManager(nc_file_path=obs_data_path)


    start_year = 1980
    end_year = 2010


    do_spatial_plots = False
    do_areaavg_plots = True




    season_to_months = OrderedDict()


    selected_months = [1, 2, 3, 4]

    for i, m in enumerate(calendar.month_name[1:], 1):
        if i in selected_months:
            season_to_months[m] = [i, ]

    print(season_to_months)

    nemo_icefrac_vname = "soicecov"
    nemo_managers = OrderedDict([
        ("CRCM5_NEMO", NemoYearlyFilesManager(folder="/BIG1/huziy/CRCM5_NEMO_coupled_sim_nemo_outputs/NEMO", suffix="grid_T.nc")),
    ])




    # calculate and plot
    map = Basemap(llcrnrlon=-93, llcrnrlat=41, urcrnrlon=-73,
                  urcrnrlat=48.5, projection='lcc', lat_1=33, lat_2=45,
                  lon_0=-90, resolution='i', area_thresh=10000)



    if do_spatial_plots:
        validate_2d_maps(nemo_managers, obs_manager=obs_manager, start_year=start_year, end_year=end_year,
                         season_to_months=season_to_months, nemo_icecover_name=nemo_icefrac_vname,
                         nemo_field_level_index=0, basemap=map)


    if do_areaavg_plots:
        # starting December until April
        season_month_start = 12
        season_month_count = 5


        mask_shp_file = "data/shp/Great_lakes_coast_shape/gl_cst.shp"

        validate_areaavg_annual_max(nemo_managers, obs_manager, start_year=start_year, end_year=end_year,
                                    season_month_start=season_month_start,
                                    season_month_count=season_month_count,
                                    mask_shape_file=mask_shp_file)





def test():


    month_start = 9
    month_count = 5


    d1 = datetime(2001, 1, 1)
    assert __map_date_to_seasonyear(d1, month_start, month_count) == d1.year - 1


    d2 = datetime(2000, 11, 1)
    assert __map_date_to_seasonyear(d2, month_start, month_count) == d2.year

    d3 = datetime(2000, 8, 1)
    assert __map_date_to_seasonyear(d3, month_start, month_count) == -1


    d4 = datetime(2001, 2, 1)
    assert __map_date_to_seasonyear(d4, month_start, month_count) == -1



if __name__ == '__main__':
    main()
    # test()
