from collections import OrderedDict
from pathlib import Path

from mpl_toolkits.basemap import maskoceans

from crcm5.mh_domains import default_domains
from data.cell_manager import CellManager
from domains.grid_config import GridConfig

from netCDF4 import Dataset
import numpy as np



def is_point_ocean_outlet(i, j, current_ocean_mask):
    ni, nj = current_ocean_mask.shape
    for di in range(-1, 2):
        for dj in range(-1, 2):

            if di == 0 and dj == 0:
                continue

            i1 = i + di
            j1 = j + dj



            # Handle the point at the boundaries, avoid falsly declaring internal drainage
            if i1 < 0 or i1 >= ni:
                return True

            if j1 < 0 or j1 >= nj:
                return True

            if current_ocean_mask[i1, j1]:
                return True

    return False



def get_mask_of_non_contrib_area(grid_config, dir_file):
    """

    :param grid_config:
    :param dir_file:
    :return: 2d numpy array with 1 for non-contributing cells and 0 otherwize
    """
    assert isinstance(grid_config, GridConfig)

    with Dataset(str(dir_file)) as ds:
        lons, lats, fldr, faa, cell_area = [ds.variables[k][:] for k in ["lon", "lat", "flow_direction_value", "accumulation_area", "cell_area"]]

    the_mask = np.zeros_like(lons)

    the_mask1 = maskoceans(lons, lats, the_mask, resolution="i", inlands=False)

    suspicious_internal_draining = (~the_mask1.mask) & ((fldr <= 0) | (fldr >= 256))


    i_list, j_list = np.where(suspicious_internal_draining)

    print("retained {} gridcells".format(suspicious_internal_draining.sum()))

    # Remove the points close to the coasts
    for i, j in zip(i_list, j_list):
        if is_point_ocean_outlet(i, j, the_mask1.mask):
            suspicious_internal_draining[i, j] = False
            the_mask1[i, j] = np.ma.masked

    print("retained {} gridcells".format(suspicious_internal_draining.sum()))



    # Now get the mask upstream of the internal draining outlets
    cell_manager = CellManager(flow_dirs=fldr, lons2d=lons, lats2d=lats, accumulation_area_km2=faa)
    i_list, j_list = np.where(suspicious_internal_draining)
    for i, j in zip(i_list, j_list):
        amask = cell_manager.get_mask_of_upstream_cells_connected_with_by_indices(i, j)

        suspicious_internal_draining |= amask > 0

    return suspicious_internal_draining


def main():


    directions_dir = Path("/HOME/huziy/skynet3_rech1/directions_for_ManitobaHydro")

    # Create the directory for the shapes
    shp_dir = directions_dir.joinpath("shp_noncontributing")
    if not shp_dir.is_dir():
        shp_dir.mkdir()


    grid_config_to_dirfile = OrderedDict([
        (default_domains.bc_mh_044, directions_dir.joinpath("directions_mh_0.44deg.nc")),
        (default_domains.bc_mh_011, directions_dir.joinpath("directions_mh_0.11deg.nc")),
        (default_domains.bc_mh_022, directions_dir.joinpath("directions_mh_0.22deg.nc")),

    ])


    for gc, dir_file in grid_config_to_dirfile.items():

        out_shp_filename = "{}_noncontributing.shp".format(dir_file.name[:-3])

        the_mask = get_mask_of_non_contrib_area(gc, dir_file)

        assert isinstance(gc, GridConfig)

        # Export the cells to a shapefile
        gc.export_to_shape_fiona(shp_folder=str(shp_dir), shp_filename=out_shp_filename, export_mask=the_mask)




if __name__ == '__main__':
    main()