import os
from rpn.rpn import RPN
from util import direction_and_value
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'huziy'



def get_previous_ij_list(i0, j0, dirs, lake_fr, glob_lakefr_limit=0.6):
    i_list = []
    j_list = []
    nx, ny = dirs.shape
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di == 0 and dj == 0:
                continue
            i = i0 + di
            j = j0 + dj

            if i >= nx or i < 0:
                continue
            if j >= ny or j < 0:
                continue

            if lake_fr[i, j] < glob_lakefr_limit:
                continue

            i_next, j_next = direction_and_value.to_indices(i, j, dirs[i, j])

            if i_next != i0 or j_next != j0:
                continue

            i_list.append(i)
            j_list.append(j)

            i_upper, j_upper = get_previous_ij_list(i, j, dirs, lake_fr, glob_lakefr_limit=glob_lakefr_limit)
            i_list.extend(i_upper)
            j_list.extend(j_upper)

    return i_list, j_list


def get_glob_lakes_mask(dirs, lakefr, lake_outlets, glob_lakefr_limit=0.6):
    lkou = np.round(lake_outlets).astype(int)

    i_out, j_out = np.where(lkou == 1)

    lakes_mask = np.zeros(dirs.shape) - 1
    lake_id = 0
    for i0, j0 in zip(i_out, j_out):

        i_prev, j_prev = get_previous_ij_list(i0, j0, dirs, lake_fr=lakefr, glob_lakefr_limit=glob_lakefr_limit)
        i_prev = np.array([i0, ] + i_prev)
        j_prev = np.array([j0, ] + j_prev)
        print(len(j_prev))

        if np.any(lakes_mask[i_prev, j_prev] >= 0):
            print("Overlap error")

        lakes_mask[i_prev, j_prev] = lake_id
        print("lake id = {0}, ncells = {1}".format(lake_id, len(i_prev)))
        lake_id += 1

    return lakes_mask


def main():
    folder = "/home/huziy/skynet3_rech1/geof_lake_infl_exp"
    fName = "geophys_Quebec_0.1deg_260x260_with_dd_v6"
    path = os.path.join(folder, fName)

    rObj = RPN(path)

    glob_lakefr_limit = 0.6
    lkou = rObj.get_first_record_for_name("LKOU")[7:-7, 7:-7]
    print("lkou(min-max):", lkou.min(), lkou.max())
    print("n_outlets = {0}".format(lkou.sum()))

    lkfr = rObj.get_first_record_for_name("LKFR")[7:-7, 7:-7]
    print("lkfr(min-max):", lkfr.min(), lkfr.max())

    dirs = rObj.get_first_record_for_name("FLDR")[7:-7, 7:-7]
    print("fldr(min-max):", dirs.min(), dirs.max())

    rObj.close()

    lakes_mask = get_glob_lakes_mask(dirs, lakefr=lkfr, lake_outlets=lkou, glob_lakefr_limit=glob_lakefr_limit)

    lakes_mask = np.ma.masked_where(lakes_mask < 0, lakes_mask)
    plt.pcolormesh(lakes_mask.transpose())
    plt.colorbar()

    plt.figure()
    plt.pcolormesh(np.ma.masked_where(lkfr >= 0.6, lkfr).transpose())

    plt.show()


if __name__ == "__main__":
    import application_properties

    application_properties.set_current_directory()
    main()
    print("Hello world")
