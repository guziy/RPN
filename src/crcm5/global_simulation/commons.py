from collections import OrderedDict

from rpn.domains import lat_lon
from scipy.spatial import cKDTree
import numpy as np

default_seasons = OrderedDict([
    ("Winter", (1, 2, 12)),
    ("Spring", (3, 4, 5)),
    ("Summer", (6, 7, 8)),
    ("Fall", (9, 10, 11)),
])

var_name_to_file_prefix = {
    "TT": "dm", "PR": "pm"
}


def interpolate_to_uniform_global_grid(data_in, lons_in, lats_in, out_dx=0.5):
    """
    Interpolate data to a regular, global latlon grid
    :param data_in:
    :param lons_in:
    :param lats_in:
    :param out_dx:
    :return:
    """
    x, y, z = lat_lon.lon_lat_to_cartesian(lons_in.flatten(), lats_in.flatten())
    tree = cKDTree(list(zip(x, y, z)))

    lons_out = np.arange(-180, 180, 0.5)
    lats_out = np.arange(-90, 90, 0.5)

    lats_out, lons_out = np.meshgrid(lats_out, lons_out)

    x_out, y_out, z_out = lat_lon.lon_lat_to_cartesian(lons_out.flatten(), lats_out.flatten())

    dists, inds = tree.query(list(zip(x_out, y_out, z_out)))

    data_out = data_in.flatten()[inds].reshape(lons_out.shape)

    return lons_out, lats_out, data_out
