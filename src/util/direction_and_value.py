__author__ = "huziy"
__date__ = "$Jul 31, 2011 11:59:38 PM$"

import numpy as np

iShifts = np.array([1, 1, 0, -1, -1, -1, 0, 1])
jShifts = np.array([0, -1, -1, -1, 0, 1, 1, 1])
values = np.array([1, 2, 4, 8, 16, 32, 64, 128])



def get_shifts_from_direction_matrix(dir_mat):

    where_valid_dirs = (dir_mat > 0) & (dir_mat <= 128)
    valid_dirs = dir_mat[where_valid_dirs]


    i_shifts = -2 * np.ones_like(dir_mat)
    j_shifts = -2 * np.ones_like(dir_mat)


    i_shifts[where_valid_dirs] = iShifts[np.log2(valid_dirs).astype(int)]
    j_shifts[where_valid_dirs] = jShifts[np.log2(valid_dirs).astype(int)]

    return i_shifts.astype(int), j_shifts.astype(int)


# Converts direction to value and vice versa

def to_value(i, j, iNext, jNext):
    di = iNext - i
    dj = jNext - j

    if iNext < 0 or jNext < 0:
        return -1

    for v, iShift, jShift in zip(values, iShifts, jShifts):
        if di == iShift and dj == jShift:
            return v
    return -1


def to_indices(i, j, value):
    if value < 0:
        return -1, -1
    for v, iShift, jShift in zip(values, iShifts, jShifts):
        if value == v:
            return i + iShift, j + jShift
    return -1, -1


def flowdir_values_to_shift(flowdir_values):
    i_shift = np.array(iShifts)
    j_shift = np.array(jShifts)

    indices = np.ones(flowdir_values.shape, dtype=int)
    for i, v in enumerate(values):
        sel = v == flowdir_values
        if np.any(sel):
            indices[sel] = i

    return i_shift[indices], j_shift[indices]


if __name__ == "__main__":
    print("Hello World")
  