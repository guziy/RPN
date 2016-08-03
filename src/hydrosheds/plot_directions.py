from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from domains import grid_config
import numpy as np
from domains.grid_config import GridConfig
from util import direction_and_value
from util.geo.mask_from_shp import get_mask

__author__ = 'huziy'

params = dict(
        dx=0.1, dy=0.1,
        lon1=180, lat1=0.0,
        lon2=-84, lat2=1.0,
        iref=105, jref=100,
        ni=210, nj=130,
        xref=276.0, yref=48.0
)

gc = GridConfig(**params)


def plot_directions(nc_path_to_directions="", grid_config=gc, margin=20, shape_path_to_focus_polygons=None):
    """

    :param margin: margin=20 corresponds to the usual free zone
    :param grid_config:
    :param nc_path_to_directions:
    :param shape_path_to_focus_polygons: if Not None, the path to the polygons
        Everything outside the specified polygons is masked

    """

    fig = plt.figure(figsize=(15, 15))

    ds = Dataset(nc_path_to_directions)




    var_name = "accumulation_area"
    data = ds.variables[var_name][margin:-margin, margin:-margin]

    data = np.ma.masked_where(data <= 0, data)

    # flow directions
    fldr = ds.variables["flow_direction_value"][margin:-margin, margin:-margin]

    i_shifts, j_shifts = direction_and_value.flowdir_values_to_shift(fldr)

    lons, lats = [ds.variables[key][margin:-margin, margin:-margin] for key in ["lon", "lat"]]

    reg_of_interest = None
    if shape_path_to_focus_polygons is not None:
        reg_of_interest = get_mask(lons, lats, shp_path=shape_path_to_focus_polygons) > 0

        i_list, j_list = np.where(reg_of_interest)

        mask_margin = 5  # margin of the mask in grid points
        i_min = min(i_list) - mask_margin
        i_max = max(i_list) + mask_margin

        j_min = min(j_list) - mask_margin
        j_max = max(j_list) + mask_margin

        bsmap = grid_config.get_basemap(lons=lons[i_min:i_max + 1, j_min:j_max + 1], lats=lats[i_min:i_max + 1, j_min:j_max + 1])

        assert isinstance(bsmap, Basemap)
        bsmap.readshapefile(shapefile=shape_path_to_focus_polygons[:-4], name="basin", linewidth=1, color="m")

    else:
        bsmap = grid_config.get_basemap(lons=lons, lats=lats)

    x, y = bsmap(lons, lats)


    nx, ny = x.shape
    inds_j, inds_i = np.meshgrid(range(ny), range(nx))
    inds_i_next = inds_i + i_shifts
    inds_j_next = inds_j + j_shifts

    inds_i_next = np.ma.masked_where((inds_i_next == nx) | (inds_i_next == -1), inds_i_next)
    inds_j_next = np.ma.masked_where((inds_j_next == ny) | (inds_j_next == -1), inds_j_next)

    u = np.ma.masked_all_like(x)
    v = np.ma.masked_all_like(x)

    good = (~inds_i_next.mask) & (~inds_j_next.mask)
    u[good] = x[inds_i_next[good], inds_j_next[good]] - x[inds_i[good], inds_j[good]]
    v[good] = y[inds_i_next[good], inds_j_next[good]] - y[inds_i[good], inds_j[good]]

    bsmap.fillcontinents(color='0.2', lake_color='aqua')
    bsmap.drawmapboundary(fill_color="aqua")

    bsmap.quiver(x, y, u, v,
                 pivot="tail", width=0.0005, scale_units="xy", headlength=20, headwidth=15, scale=1, zorder=5)

    bsmap.drawcoastlines(linewidth=0.5)

    bsmap.drawrivers(color="b")


    plt.savefig(nc_path_to_directions[:-3] + ".png", bbox_inches="tight")




if __name__ == "__main__":
    # main()
    plot_directions(nc_path_to_directions="/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_great_lakes_210_130_0.1deg_v2.nc")
