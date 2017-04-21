from functools import lru_cache
import numpy as np
from scipy.spatial import KDTree

from lake_effect_snow import common_params
from util.geo import lat_lon


import matplotlib.pyplot as plt



def get_nonlocal_mean_snowfall(lons, lats, region_of_interest, kdtree, snowfall, lake_mask, outer_radius_km=500):

    nonlocal_snfl = snowfall.copy()

    # need non-local snowfall only for points where actual snowfall occurs
    where_snows_heavy = nonlocal_snfl.sum(dim="t").values >= common_params.lower_limit_of_daily_snowfall

    for i, j in zip(*np.where(region_of_interest & where_snows_heavy)):
        lon, lat = lons[i, j], lats[i, j]

        the_mask = get_non_local_mask_for_location(lon, lat, kdtree, mask_shape=lons.shape, outer_radius_km=outer_radius_km)

        # ignore lakes and the areas within the lake effect zone
        the_mask &= (~region_of_interest)
        the_mask &= (~lake_mask)



        # print("Size of the nonlocal region is {} cells".format(the_mask.sum()))

        if np.any(the_mask):
            for t_ind, snfl_current in enumerate(snowfall):
                nonlocal_snfl[t_ind, i, j] = snfl_current.values[the_mask].mean()
        else:
            nonlocal_snfl[:, i, j] = snowfall[:, i, j]

    return nonlocal_snfl



def get_non_local_mask_for_location(lon0, lat0, ktree: KDTree, mask_shape=None, outer_radius_km=500):

    x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lon0, lat0, R=lat_lon.EARTH_RADIUS_METERS)

    METERS_PER_KM = 1000.0

    npoints = len(ktree.data)

    assert npoints == np.product(mask_shape)

    # dists, inds = ktree.query((x0, y0, z0), k=npoints, distance_upper_bound=outer_radius_km * 1000.0)
    dists, inds = ktree.query((x0, y0, z0), k=npoints)

    result = np.zeros(mask_shape, dtype=bool).flatten()

    # result[inds] = True
    result[inds] = (dists <= outer_radius_km * METERS_PER_KM)
    result.shape = mask_shape

    return result




if __name__ == '__main__':
    lons = np.zeros((2,2))
    lats = np.zeros((2,2))



    xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons.flatten(), lats.flatten())

    ktree = KDTree(list(zip(xs, ys, zs)))


    print(len(ktree.data))



    the_mask = get_non_local_mask_for_location(5, 0, ktree, mask_shape=(2, 2))
    print(the_mask)
