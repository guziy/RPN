
import numpy as np


def get_rectangular_region_from_mask_and_margin(mask, margin_points=5):
    """
    Mask is not always rectangular, make the region rectangular and add the margin
    :param mask:
    :param margin_points:
    :return:
    """
    i_list, j_list = np.where(mask)
    imin, imax = np.min(i_list) - margin_points, np.max(i_list) + margin_points
    jmin, jmax = np.min(j_list) - margin_points, np.max(j_list) + margin_points

    return imin, imax, jmin, jmax
