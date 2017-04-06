from pathlib import Path

from mpl_toolkits.basemap import Basemap
from scipy.spatial import KDTree

from lake_effect_snow import common_params

import xarray as xr

from lake_effect_snow.lake_effect_snowfall_entry import get_zone_around_lakes_mask
from util import plot_utils
from util.geo import lat_lon
from util.geo.mask_from_shp import get_mask

import matplotlib.pyplot as plt


def main(path="", reg_of_interest=None):

    out_folder = Path(path).parent



    clevs = common_params.clevs_lkeff_snowfall


    ds = xr.open_dataset(path)
    snfl = ds["snow_fall"].squeeze()
    lons, lats = snfl.coords["lon"].values, snfl.coords["lat"].values


    near_lake_100km_zone_mask = None

    if reg_of_interest is None:

        reg_of_interest = common_params.great_lakes_limits.get_mask_for_coords(lons, lats)

        # temporary
        lake_mask = get_mask(lons, lats, shp_path=common_params.GL_COAST_SHP_PATH) > 0.1
        print("lake_mask shape", lake_mask.shape)

        # mask lake points
        reg_of_interest &= ~lake_mask

        # get the KDTree for interpolation purposes
        ktree = KDTree(data=list(zip(*lat_lon.lon_lat_to_cartesian(lon=lons.flatten(), lat=lats.flatten()))))

        # define the 100km near lake zone
        near_lake_100km_zone_mask = get_zone_around_lakes_mask(lons=lons, lats=lats, lake_mask=lake_mask,
                                                              ktree=ktree, dist_km=100)

        reg_of_interest &= near_lake_100km_zone_mask


    # snfl.plot()
    # plt.show()

    b = Basemap(lon_0=180,
                llcrnrlon=common_params.great_lakes_limits.lon_min,
                llcrnrlat=common_params.great_lakes_limits.lat_min,
                urcrnrlon=common_params.great_lakes_limits.lon_max,
                urcrnrlat=common_params.great_lakes_limits.lat_max,
                resolution="i")

    xx, yy = b(lons, lats)


    # print("Basemap corners: ", lons[i_min, j_min] - 360, lons[i_max, j_max] - 360)

    plot_utils.apply_plot_params(font_size=20)
    fig = plt.figure()

    nrows = 1
    ncols = 1
    gs = GridSpec(ncols=ncols, nrows=nrows)

    # bn = BoundaryNorm(clevs, len(clevs) - 1)
    # cmap = cm.get_cmap("nipy_spectral")

    cmap, bn = colors.from_levels_and_colors(clevs, ["white", "indigo", "blue", "dodgerblue", "aqua", "lime", "yellow", "gold",
                                                     "orange", "red"])
