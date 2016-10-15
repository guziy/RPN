from datetime import timedelta

from scipy.spatial import cKDTree as KDTree
from util.geo import lat_lon
import numpy as np
from geopy.distance import distance


import multiprocessing

# maximum number of iterations for backtracking
N_ITER_MAX_BACKTRACK = 10


def get_velocity_at(vel_field, r, ktree, i_grd, j_grd, nneighbours=1):
    """
    :type ktree: KDTree
    :param j_grd:
    :param i_grd:
    :param ktree:
    :param vel_field: 2d field of 3D velocity (3, x, y)
    :param r: vector where the value is needed
    """

    nprocs = max(1, multiprocessing.cpu_count() // 2)

    if nneighbours == 1:
        dist, ind_r = ktree.query(r, k=nneighbours, n_jobs=nprocs)
        i, j = i_grd.flatten()[ind_r], j_grd.flatten()[ind_r]
        return vel_field[:, i, j]
    else:
        dists, inds_r = ktree.query(r, k=nneighbours, n_jobs=nprocs)

        i_arr, j_arr = i_grd.flatten()[inds_r], j_grd.flatten()[inds_r]

        w_total = 0.0
        vel_mean = None
        for i, j, dist in zip(i_arr, j_arr, dists):
            if dist == 0:
                return vel_field[:, i, j]

            wi = 1.0 / dist
            w_total += wi

            if vel_mean is None:
                vel_mean = wi * vel_field[:, i, j]
            else:
                vel_mean += wi * vel_field[:, i, j]

        return vel_mean / w_total


def get_epsilon(lons2d, lats2d):
    """
    Get the minimum displacement of the starting point to be able to declare convergence
    :param lons2d:
    :param lats2d:
    :return:
    """

    eps = distance((lats2d[1, 1], lons2d[1, 1]), (lats2d[0, 0], lons2d[0, 0])).meters
    return eps


def get_wind_blows_from_lakes_mask(lons, lats, u_we, v_sn, lake_mask, ktree, region_of_interest=None, dt_secs=None,
                                   nneighbours=1):
    """
    Get masks of the regions where wind is blowing from lakes

    :param nneighbours: number of closest neighbours to consider for wind interpolation
    :param region_of_interest:
    :param dt_secs: time step of the wind fields (if not specified, assume 1 day)
    :param u_we: dimensions (time, x, y)
    :param v_sn: same as for u_we
    :param lake_mask: dimensions (x, y)
    :param ktree: Needed for passing from the physical space to the index space, created from flattened lons, last and
    represented in the 3D cartesian space with an origin at the centre of the Earth.
    """

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

    nprocs = max(1, multiprocessing.cpu_count() // 2)

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

        converged_count = 0
        for xa, ya, za in zip(xa_list, ya_list, za_list):

            r0 = np.array([xa, ya, za])

            r_prev = np.zeros_like(r0)

            dist, ind_r0 = ktree.query(r0, n_jobs=nprocs)
            i_r0, j_r0 = i_grid.flatten()[ind_r0], j_grid.flatten()[ind_r0]


            vel_t_r0 = get_velocity_at(vel_t, r0, ktree=ktree, i_grd=i_grid, j_grd=j_grid, nneighbours=nneighbours)
            vel_tm1_r0 = get_velocity_at(vel_tm1, r0, ktree=ktree, i_grd=i_grid, j_grd=j_grid, nneighbours=nneighbours)

            r1 = r0 - dt_secs * 0.5 * (vel_tm1_r0 + vel_t_r0)

            # Find the departure point
            converged = False
            for it in range(N_ITER_MAX_BACKTRACK):
                if np.sum((r1 - r_prev) ** 2) ** 0.5 <= eps:
                    converged = True
                    break

                rmiddle = (r0 + r1) * 0.5

                vel_t_rmiddle = get_velocity_at(vel_t, rmiddle, ktree=ktree, i_grd=i_grid, j_grd=j_grid, nneighbours=nneighbours)
                vel_tm1_rmiddle = get_velocity_at(vel_tm1, rmiddle, ktree=ktree, i_grd=i_grid, j_grd=j_grid, nneighbours=nneighbours)

                r_prev = r1
                r1 = r0 - dt_secs * 0.5 * (vel_tm1_rmiddle + vel_t_rmiddle)

            # print a message if the iteration for the departure point has not converged
            if not converged:
                # msg = "Iterations for the departure point has not converged: delta={}, eps={}"
                # print(msg.format(np.sum((r1 - r_prev) ** 2) ** 0.5, eps))
                pass
            else:
                converged_count += 1

            dist, ind_r1 = ktree.query(r1, n_jobs=nprocs)
            i_r1, j_r1 = i_grid.flatten()[ind_r1], j_grid.flatten()[ind_r1]


            ill = min(i_r0, i_r1)
            jll = min(j_r0, j_r1)
            iur = max(i_r0, i_r1)
            jur = max(j_r0, j_r1)

            # 1 if the fetch is from lake, 0 otherwize
            fetch_from_lake_mask[ti, i_r0, j_r0] = lake_mask[ill:iur + 1, jll:jur + 1].sum() > 0.5


        print("Converged {} of {} considered points".format(converged_count, len(xa_list)))
        print("Finished {}/{} ".format(ti, nt))

        if ti == 20:
            break

    return fetch_from_lake_mask