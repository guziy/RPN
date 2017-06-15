from netCDF4 import Dataset, OrderedDict
from rpn import level_kinds
from rpn.rpn import RPN
import numpy as np
from scipy.spatial import KDTree

from util.geo import lat_lon


g_const = 9.80665

def get_closest_value_for_point(lon0, lat0, lons_s, lats_s, data_s, tree=None):

    x, y, z = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())

    if tree is None:
        tree = KDTree(data=list(zip(x, y, z)))

    x0, y0, z0 = lat_lon.lon_lat_to_cartesian(lon0, lat0)

    d, i = tree.query((x0, y0, z0))

    return data_s.flatten()[i]



def main():

    lon0, lat0 = -114.0708, 51.0486

    surf_gz_path = "/RESCUE/skynet3_rech1/huziy/hail/erainterim_0.75d/surface_geopotential.nc"
    gz_fields_path = "/HOME/data/Driving_data/Pilots/ERA-Interim_0.75/Pilots/era-interim_0.75d_198004"


    # --
    with Dataset(surf_gz_path) as ds:
        lons, lats = ds.variables["longitude"][:], ds.variables["latitude"][:]
        lats, lons = np.meshgrid(lats, lons)

        surface_elev = get_closest_value_for_point(lon0, lat0, lons, lats, ds.variables["z"][:].squeeze().T) / g_const



    # --
    with RPN(gz_fields_path) as r:
        assert isinstance(r, RPN)
        gz_data = r.get_4d_field("GZ", level_kind=level_kinds.PRESSURE)

        lons_r, lats_r = r.get_longitudes_and_latitudes_for_the_last_read_rec()


        dates = sorted(gz_data)

        lev_to_field = gz_data[dates[0]]

        levels = sorted(lev_to_field)

        pelevs = [get_closest_value_for_point(lon0, lat0, lons_r, lats_r, lev_to_field[lev]) * 10 for lev in levels]



    print(OrderedDict(list(zip(levels, pelevs))))
    print("surf elev = {}".format(surface_elev))









if __name__ == '__main__':
    main()