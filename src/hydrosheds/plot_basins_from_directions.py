from util.direction_and_value import get_shifts_from_direction_matrix

__author__ = 'huziy'

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from rpn.domains.rotated_lat_lon import RotatedLatLon
from numba import jit


# @jit("int32(int32[:,:], int32[:,:], int32[:,:], int32, int32, int32)")
def color_upstream(colors=None,
                   i_shifts=None,
                   j_shifts=None,
                   i0=None,
                   j0=None, current_color=None):
    """
    Recursive function coloring basins
    :param colors:
    :param i_shifts:
    :param j_shifts:
    :param i0:
    :param j0:
    :param current_color:
    """
    nx, ny = colors.shape

    has_cells_upstream = False
    has_next = i_shifts[i0, j0] > -2 and j_shifts[i0, j0] > -2

    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            i1 = i0 + di
            j1 = j0 + dj

            if i1 < 0 or i1 >= nx or j1 < 0 or j1 >= ny:
                continue

            if (i1 + i_shifts[i1, j1] == i0) and (j1 + j_shifts[i1, j1] == j0):
                color_upstream(colors=colors,
                               i_shifts=i_shifts, j_shifts=j_shifts,
                               i0=i1, j0=j1,
                               current_color=current_color)
                has_cells_upstream = True

    if has_cells_upstream or has_next:
        colors[i0, j0] = current_color
    return has_cells_upstream and not has_next


def get_basin_map(directions):
    i_shifts, j_shifts = get_shifts_from_direction_matrix(directions)

    result = -np.ones_like(directions)

    where_good_points = (i_shifts > -2) & (j_shifts > -2)
    where_bad_points = ~where_good_points

    i_list, j_list = np.where(where_bad_points)

    current_color = 1

    # select outlets
    for i, j in zip(i_list, j_list):
        is_outlet = color_upstream(colors=result, i_shifts=i_shifts, j_shifts=j_shifts,
                                            i0=i, j0=j, current_color=current_color)
        current_color += int(is_outlet) * 5
    result = np.ma.masked_where(result < 0, result)
    return result


def save_matrix_to_netcdf(fdv):
    path = "basins.nc"

    nx, ny = fdv.shape
    ds = Dataset(path, "w")
    ds.createDimension("lon", nx)
    ds.createDimension("lat", ny)

    v = ds.createVariable("basin_id", "i4", ("lon", "lat"))
    v[:] = fdv
    ds.close()


def main():
    # path = "/skynet3_exec2/aganji/2GW_new/guziy-water_route_offline-d4627cd00b84/infocell.nc"

    path = "/skynet3_rech1/huziy/temp/directions_north_america_0.44deg_arman.v5.nc"

    # projection definition
    rll = RotatedLatLon(lon1=-97, lat1=47.5, lon2=-7.0, lat2=0.0)

    ds = Dataset(path)
    print(list(ds.variables.keys()))

    # read data from the infocell file
    fdv = ds.variables["flow_direction_value"][:]
    lons, lats = ds.variables["lon"][:], ds.variables["lat"][:]

    # get basemap object
    bmp = rll.get_basemap_object_for_lons_lats(lons2d=lons, lats2d=lats, resolution="c")
    x, y = bmp(lons, lats)

    colors = get_basin_map(fdv)

    save_matrix_to_netcdf(colors)


    im = bmp.pcolormesh(x, y, colors)
    bmp.colorbar(im)
    bmp.drawcoastlines()
    plt.show()


if __name__ == '__main__':
    import time
    t0 = time.clock()
    main()
    print("Elapsed time: {} s".format(time.clock() - t0))