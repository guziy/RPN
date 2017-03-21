from pathlib import Path

from netCDF4 import Dataset

from application_properties import main_decorator

import numpy as np


#
from data.cell_manager import CellManager

iShifts = np.array([1, 1, 0, -1, -1, -1, 0, 1])
jShifts = np.array([0, -1, -1, -1, 0, 1, 1, 1])
values = np.array([1, 2, 4, 8, 16, 32, 64, 128])


ACCINDEX_TO_DEFINE = -15


class LightCell(object):

    def __init__(self, **kwars):
        self.next = None
        self.previous = []

        self.dirvalue = -1

        self.accindex = -1

        self.i_dir = -1

        self.i_accindex = -1



    def set_next(self, cell):
        self.next = cell

        if cell is not None:
            cell.previous.append(self)




def get_index_shifts_from_dirvalue(dvalue, origin_upper_left=True):

    if dvalue > 128 or dvalue < 1:
        return None, None

    i = int(np.log2(dvalue) + 0.5)

    xshift = iShifts[i]
    yshift = jShifts[i]

    if origin_upper_left:
        yshift *= -1

    return yshift, xshift




def get_dirvalue_from_index_shifts(yshift, xshift, origin_upper_left=True):
    mul = 1
    if origin_upper_left:
        mul = -1

    return values[(iShifts == xshift) & (jShifts == mul * yshift)][0]



def modify_direction(point_props, i_dir, dir_arr, origin_upper_left=True, lons=None, lats=None):
    """

    :param point_props:
    :param i_dir: (ilat, ilon) of the point to modify in the direction matrix space
    :param i_accind: (ilat, ilon) of the point to modify in the direction matrix space
    """

    msg = "expect {}, got {}".format(point_props["dir_old"], dir_arr[i_dir])

    assert point_props["dir_old"] == dir_arr[i_dir], msg
    # assert point_props["acc"] == accindex_arr[i_accindex]

    # modify the direction and accumulation index downstream
    if lons is None or lats is None:
        print("{}: {} --> {}".format(i_dir, dir_arr[i_dir], point_props["dir_new"]))
    else:
        print("(lat,lon) = {}: {} --> {}".format((lats[i_dir[0]], lons[i_dir[1]]), dir_arr[i_dir], point_props["dir_new"]))

    dir_arr[i_dir] = point_props["dir_new"]







def get_point_indices(lon1, lat1, lons_1d, lats_1d, origin_upper_left=True):
    assert origin_upper_left
    ilon = np.argmin(np.abs(lons_1d - lon1))
    ilat = np.argmin(np.abs(lats_1d - lat1))
    return ilon, ilat



def save_array_to_nc_file(ds_stamp, ds_out, new_data, varname_to_modify, rename=None):
    # Copy dimensions
    for dname, the_dim in ds_stamp.dimensions.items():
        print(dname, len(the_dim))
        ds_out.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)


    # Copy variables
    for the_varname, varin in ds_stamp.variables.items():

        if the_varname == varname_to_modify:

            if rename is not None:
                the_varname = rename[varname_to_modify]

            outVar = ds_out.createVariable(the_varname, "i4", varin.dimensions)

            # Copy variable attributes
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})

            outVar[:] = new_data
        else:
            outVar = ds_out.createVariable(the_varname, varin.datatype, varin.dimensions)
            print(varin.datatype, the_varname)

            # Copy variable attributes
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})

            outVar[:] = varin[:]




def calculate_acc_index_for_point(i, j, dir_arr, cache, origin_upper_left=True):

    if cache[i, j] > 0:
        return cache[i, j]

    n1, n2 = dir_arr.shape

    res = 1
    for dj in range(-1, 2):
        for di in range(-1, 2):

            if di == 0 and dj == 0:
                continue

            i1 = i + di
            j1 = j + dj

            if i1 < 0 or j1 < 0:
                continue

            if i1 >= n1 or j1 >= n2:
                continue


            # skip if the point is masked
            if dir_arr.mask[i1, j1]:
                continue

            shift = get_index_shifts_from_dirvalue(dir_arr[i1, j1], origin_upper_left=origin_upper_left)

            if None in shift:
                continue


            if (i1 + shift[0], j1 + shift[1]) == (i, j):
                res += calculate_acc_index_for_point(i1, j1, dir_arr, cache, origin_upper_left=origin_upper_left)

    cache[i, j] = res
    return res


def calculate_acc_index(dir_arr, origin_upper_left=True):

    # modify maximum recursion depth if required
    import sys
    rec_depth = sys.getrecursionlimit()
    sys.setrecursionlimit(max(dir_arr.shape[0] * dir_arr.shape[1], rec_depth))



    acc_index = np.ones(dir_arr.shape)
    cache = -np.ones(dir_arr.shape)




    n1, n2 = acc_index.shape

    for i in range(n1):
        if i % 100 == 0:
            print("{}/{} ...".format(i, n1))

        for j in range(n2):
            acc_index[i, j] = calculate_acc_index_for_point(i, j, dir_arr, cache, origin_upper_left=origin_upper_left)

    # print(acc_index.min(), acc_index.max())

    return acc_index





def test_calculate_acc_index():


    dirs = [
        [ 2,  4,   4,  1, 128],
        [ 1,  1,   1, 64,  64],
        [ 1,  1,   1, 64,  32],
        [ 1,  1,   1, 64,  16],
        [64, 64, 128, 64,  32]
    ]


    dirs = np.ma.array(dirs)

    accs = calculate_acc_index(dirs, origin_upper_left=True)


    import matplotlib.pyplot as plt
    plt.figure()

    im = plt.pcolormesh(np.flipud(accs))
    plt.colorbar(im)
    plt.show()


    print(accs)







@main_decorator
def main():
    """
    Note assume lat to be the first dimension and the (0,0) gridcell in the upper left corner
    """
    in_directions_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/data/netcdf/Sasha_hydrosheds_continents_and_global_raster_and_nc_30s/glob_30s/dir_30s.nc"
    in_directions_vname = "dir"

    # in_accindex_path = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/data/netcdf/NA/na_acc_30s.nc"
    # in_accindex_vname = "flow_accumulation"


    out_folder = Path("hydrosheds_corrected_mh_glob/")
    if not out_folder.is_dir():
        out_folder.mkdir()


    origin_upper_left = True

    # change this for other correction sets
    from crcm5.mh_domains.hs_correction_sets.rat_river_correction import to_invert, to_modify



    # get the initial fields of directions and indices
    with Dataset(in_directions_path) as ds:
        dirs_i = ds.variables[in_directions_vname][:]
        lons_dir = ds.variables["lon"][:]
        lats_dir = ds.variables["lat"][:]



    # allocate the result arrays
    dirs_new = dirs_i.copy()


    # simple point directions to modify
    for p_id, p_props in to_modify.items():
        print("Working on {}".format(p_id))

        lon, lat = [p_props[k] for k in ["lon", "lat"]]

        ilon_dir, ilat_dir = get_point_indices(lon, lat, lons_1d=lons_dir, lats_1d=lats_dir)

        modify_direction(p_props, (ilat_dir, ilon_dir), dirs_new, lons=lons_dir, lats=lats_dir)






    # populate the stack and then set the directions from to to bottom
    #
    ind_dir_stack = []
    i_shift_stack = []
    for p_id, p_props in to_invert.items():
        print("Reversing {}".format(p_id))

        # get indices for the entry point in order to know when to stop
        enter_point = p_props["enterpoint"]
        lon_enter, lat_enter = [enter_point[k] for k in ["lon", "lat"]]
        ilon_dir_enter, ilat_dir_enter = get_point_indices(lon_enter, lat_enter, lons_1d=lons_dir, lats_1d=lats_dir)


        # start point of the way
        lon, lat = [p_props[k] for k in ["lon", "lat"]]

        ilon_dir, ilat_dir = get_point_indices(lon, lat, lons_1d=lons_dir, lats_1d=lats_dir)

        # coordinates in the direction field and in the accumulation index fields for the exit point
        i_dir_exit = (ilat_dir, ilon_dir)


        i_dir_current = (ilat_dir, ilon_dir)
        i_shift_old = get_index_shifts_from_dirvalue(dirs_new[i_dir_current], origin_upper_left=origin_upper_left)


        # --- fill in the stacks with the indices of the cells to modify
        ind_dir_stack.append(i_dir_current)
        i_shift_stack.append(None)

        safe_count = 0
        while ind_dir_stack[-1] != (ilat_dir_enter, ilon_dir_enter):


            the_top = tuple(ind_dir_stack[-1][k] + i_shift_old[k] for k in [0, 1])
            ind_dir_stack.append(the_top)



            i_shift_stack.append(i_shift_old)


            # print(ind_dir_stack[-1])
            # print("dir = {}".format(dirs_new[ind_dir_stack[-1]]))
            i_shift_old = get_index_shifts_from_dirvalue(dirs_new[ind_dir_stack[-1]], origin_upper_left=origin_upper_left)


            # to avoid infinite loops
            safe_count += 1
            if safe_count >= 1000:
                raise Exception("Infinite loop condition...")



        # revert the directions
        for the_ind_dir, the_i_shift_old in zip(reversed(ind_dir_stack), reversed(i_shift_stack)):

            if the_i_shift_old is None:
                continue

            the_path_point_props = {
                "dir_old": dirs_new[the_ind_dir],
                "dir_new": get_dirvalue_from_index_shifts(*[-di for di in the_i_shift_old], origin_upper_left=origin_upper_left),
            }

            modify_direction(the_path_point_props, the_ind_dir, dirs_new, lons=lons_dir, lats=lats_dir)


        # Adjust directions downstream of the exit point as well
        the_ind_dir = i_dir_exit
        for next_dirvalue in p_props["dir_new"]:
            modify_direction({"dir_new": next_dirvalue, "dir_old": dirs_new[the_ind_dir]},
                             the_ind_dir, dirs_new, lons=lons_dir, lats=lats_dir)

            i_shift = get_index_shifts_from_dirvalue(next_dirvalue, origin_upper_left=origin_upper_left)

            the_ind_dir = tuple(the_ind_dir[k] + i_shift[k] for k in [0, 1])


    # save the new directions to file
    out_dir_file_path = out_folder.joinpath(Path(in_directions_path).name)
    with Dataset(str(out_dir_file_path), "w") as ds:
        save_array_to_nc_file(Dataset(in_directions_path), ds, dirs_new, in_directions_vname)


    if True:
        raise Exception

    # save the new accumulation index to file

    acc_index = calculate_acc_index(dirs_new, origin_upper_left=origin_upper_left)


    out_accind_file_path = out_folder.joinpath("na_acc_30s.nc")
    with Dataset(str(out_accind_file_path), "w") as ds:
         save_array_to_nc_file(Dataset(in_directions_path), ds, acc_index, in_directions_vname,
                               rename={in_directions_vname: "flow_accumulation"})




if __name__ == '__main__':
    import time
    t0 = time.clock()
    main()
    # test_calculate_acc_index()
    print("execution time: {}".format(time.clock() - t0))