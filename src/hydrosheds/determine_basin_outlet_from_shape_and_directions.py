from mpl_toolkits.basemap import Basemap
from pathlib import Path

from application_properties import main_decorator
from util.geo.mask_from_shp import get_mask
from netCDF4 import Dataset
from crcm5.mh_domains import default_domains
import numpy as np
import matplotlib.pyplot as plt


img_folder = Path("mh")


def get_basin_outlet_indices(lons2d, lats2d, accumulation_area, shp_path=""):
    the_mask = get_mask(lons2d=lons2d, lats2d=lats2d, shp_path=shp_path)

    basin_ids = np.unique(the_mask[the_mask > 0])
    print("basin_ids = ", basin_ids)

    i_out_list = []
    j_out_list = []
    for the_id in list(basin_ids):
        vmax = np.max(accumulation_area[the_mask == the_id])
        i1, j1 = np.where(accumulation_area == vmax)
        i_out_list.append(i1)
        j_out_list.append(j1)

    i_out_list = np.array(i_out_list)
    j_out_list = np.array(j_out_list)

    return i_out_list, j_out_list


@main_decorator
def main():
    directions_file = "/RESCUE/skynet3_rech1/huziy/Netbeans Projects/Java/DDM/directions_bc-mh_0.44deg.nc"
    shape_file = default_domains.MH_BASINS_PATH


    grid_config = default_domains.bc_mh_044

    with Dataset(directions_file) as ds:
        lons, lats, fldr, acc_area = [ds.variables[k][:] for k in ["lon", "lat", "flow_direction_value", "accumulation_area"]]

    bmp, region_of_interest = grid_config.get_basemap_using_shape_with_polygons_of_interest(lons=lons, lats=lats,
                                                                                            shp_path=shape_file)


    i_out_list, j_out_list = get_basin_outlet_indices(lons2d=lons, lats2d=lats, accumulation_area=acc_area, shp_path=shape_file)
    xx, yy = bmp(lons[i_out_list, j_out_list], lats[i_out_list, j_out_list])

    fig = plt.figure()

    assert isinstance(bmp, Basemap)
    bmp.scatter(xx, yy, s=40)
    bmp.drawcoastlines()
    bmp.readshapefile(shapefile=shape_file[:4], name="basins", color="m", linewidth=2)

    fig.savefig(img_folder.joinpath("outlets_{}.png".format(grid_config.dx, )))



if __name__ == '__main__':
    main()