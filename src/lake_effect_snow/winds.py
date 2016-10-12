from datetime import timedelta

from scipy.spatial import KDTree
from util.geo import lat_lon
import numpy as np
from geopy.distance import distance

# maximum number of iterations for backtracking
N_ITER_MAX_BACKTRACK = 10


def get_velocity_at(vel_field, r, ktree, i_grd, j_grd):
    """
    :param j_grd:
    :param i_grd:
    :param ktree:
    :param vel_field: 2d field of 3D velocity (3, x, y)
    :param r: vector where the value is needed
    """

    dist, ind_r = ktree.query(r)
    i, j = i_grd.flatten()[ind_r], j_grd.flatten()[ind_r]
    return vel_field[:, i, j]


def get_epsilon(lons2d, lats2d):
    """
    Get the minimum displacement of the starting point to be able to declare convergence
    :param lons2d:
    :param lats2d:
    :return:
    """
    total_x_bottom = distance((lats2d[0, 0], lons2d[0, 0]), (lats2d[-1, 0], lons2d[-1, 0]))
    total_x_top = distance((lats2d[0, -1], lons2d[0, -1]), (lats2d[-1, -1], lons2d[-1, -1]))

    total_x = 0.5 * (total_x_bottom + total_x_top)


    total_y_left = distance((lats2d[0, 0], lons2d[0, 0]), (lats2d[0, -1], lons2d[0, -1]))
    total_y_right = distance((lats2d[-1, 0], lons2d[-1, 0]), (lats2d[-1, -1], lons2d[-1, -1]))

    total_y = 0.5 * (total_y_left + total_y_right)

    nx, ny = lons2d.shape

    # Avoid division by 0
    if nx == 1:
        nx = 2

    if ny == 1:
        ny = 2

    eps = min(total_x / (nx - 1), total_y / (ny - 1))
    return eps


def get_wind_blows_from_lakes_mask(lons, lats, u_we, v_sn, lake_mask, ktree, region_of_interest=None, dt_secs=None):
    """
    Get masks of the regions where wind is blowing from lakes

    :param region_of_interest:
    :param dt_secs: time step of the wind fields (if not specified, assume 1 day)
    :param u_we: dimensions (time, x, y)
    :param v_sn: same as for u_we
    :param lake_mask: dimensions (x, y)
    :param ktree: Needed for passing from the physical space to the index space, created from flattened lons, last and
    represented in the 3D cartesian space with an origin at the centre of the Earth.
    """

    assert isinstance(ktree, KDTree)

    if dt_secs is None:
        dt_secs = timedelta(days=1).total_seconds()

    possible_arrival_points = region_of_interest & (~lake_mask)

    nt = u_we.shape[0]


    lons_rad, lats_rad = np.radians(lons), np.radians(lats)

    # velocity shape: (3, t, x, y)
    velocity = lat_lon.geo_uv_to_cartesian_velocity(u_we=u_we, v_sn=v_sn, lons_rad=lons_rad, lats_rad=lats_rad)

    # check limiting (epsilon) distance
    eps = get_epsilon(lons, lats)
    print("epsilon = {}".format(eps))

    xa_list, ya_list, za_list = lat_lon.lon_lat_to_cartesian(lons[possible_arrival_points], lats[possible_arrival_points])


    i_grid, j_grid = np.indices(lons.shape)

    fetch_from_lake_mask = np.zeros_like(u_we, dtype=bool)

    print("Start looking if wind blows from lakes for all time steps")
    print("Number of points of interest: {}".format(np.sum(region_of_interest)))
    for ti in range(nt):

        #  get the velocity fields for t and t-dt
        vel_t = velocity[:, ti, :, :]

        if ti == 0:
            vel_tm1 = vel_t
        else:
            vel_tm1 = velocity[:, ti - 1, :, :]


        for xa, ya, za in zip(xa_list, ya_list, za_list):

            r0 = np.array([xa, ya, za])

            r_prev = np.zeros_like(r0)

            dist, ind_r0 = ktree.query(r0)
            i_r0, j_r0 = i_grid.flatten()[ind_r0], j_grid.flatten()[ind_r0]


            vel_t_r0 = get_velocity_at(vel_t, r0, ktree=ktree, i_grd=i_grid, j_grd=j_grid)
            vel_tm1_r0 = get_velocity_at(vel_tm1, r0, ktree=ktree, i_grd=i_grid, j_grd=j_grid)

            r1 = r0 - dt_secs * 0.5 * (vel_tm1_r0 + vel_t_r0)

            # Find the departure point
            converged = False
            for it in range(N_ITER_MAX_BACKTRACK):
                if np.sum((r1 - r_prev) ** 2) ** 0.5 <= eps:
                    converged = True
                    break

                rmiddle = (r0 + r1) * 0.5

                vel_t_rmiddle = get_velocity_at(vel_t, rmiddle, ktree=ktree, i_grd=i_grid, j_grd=j_grid)
                vel_tm1_rmiddle = get_velocity_at(vel_tm1, rmiddle, ktree=ktree, i_grd=i_grid, j_grd=j_grid)

                r_prev = r1
                r1 = r0 - dt_secs * 0.5 * (vel_tm1_rmiddle + vel_t_rmiddle)

            # print a message if the iteration for the departure point has not converged
            if not converged:
                msg = "Iterations for the departure point has not converged: delta={}, eps={}"
                print(msg.format(np.sum((r1 - r_prev) ** 2) ** 0.5, eps))

            dist, ind_r1 = ktree.query(r1)
            i_r1, j_r1 = i_grid.flatten()[ind_r1], j_grid.flatten()[ind_r1]


            ill = min(i_r0, i_r1)
            jll = min(j_r0, j_r1)
            iur = max(i_r0, i_r1)
            jur = max(j_r0, j_r1)

            # 1 if the fetch is from lake, 0 otherwize
            fetch_from_lake_mask[ti, i_r0, j_r0] = lake_mask[ill:iur + 1, jll:jur + 1].sum() > 0.5


        print("Finished {}/{} ".format(ti, nt))

    return fetch_from_lake_mask