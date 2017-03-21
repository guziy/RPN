


from netCDF4 import Dataset
from scipy.spatial import KDTree

from domains.grid_config import GridConfig
from nemo.generate_grid.nemo_domain_properties import known_domains
from util.geo import lat_lon
import numpy as np


def interpolate_data_nn_2d(field_in, map_indexes, target_shape):
    return field_in.flatten()[map_indexes].reshape(target_shape)


def interpolate_data_nn(field_in, map_indexes, target_shape):
    in_shape = field_in.shape

    print(in_shape)

    if len(in_shape) == 2:
        return interpolate_data_nn_2d(field_in, map_indexes, target_shape)
    else:

        data_list = []
        for i in range(in_shape[0]):
            the_field = interpolate_data_nn(field_in[i], map_indexes, target_shape)
            data_list.append(the_field)
        return np.array(data_list)


def main(in_file="", out_file=None, target_grid: GridConfig=None):


    if out_file is None:
        out_file = "{}_interpolated".format(in_file)

    # input file
    dsin = Dataset(in_file)

    # output file
    with Dataset(out_file, "w") as dsout:
        # Copy dimensions
        for dname, the_dim in dsin.dimensions.items():
            print(dname, len(the_dim))


            # change the x and y dimensions only
            if dname not in ["x", "y"]:
                dsout.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)
            elif dname == "x":
                dsout.createDimension(dname, target_grid.ni)
            elif dname == "y":
                dsout.createDimension(dname, target_grid.nj)


        lons_t, lats_t = [field.T for field in target_grid.get_lons_and_lats_of_gridpoint_centers()]
        lons_s, lats_s = [dsin.variables[k][:] for k in ["nav_lon", "nav_lat"]]

        xs, ys, zs = lat_lon.lon_lat_to_cartesian(lons_s.flatten(), lats_s.flatten())
        xt, yt, zt = lat_lon.lon_lat_to_cartesian(lons_t.flatten(), lats_t.flatten())
        ktree = KDTree(list(zip(xs, ys, zs)))
        dists, inds = ktree.query(list(zip(xt, yt, zt)))



        # Copy variables
        for v_name, varin in dsin.variables.items():

            outVar = dsout.createVariable(v_name, varin.datatype, varin.dimensions)
            print(varin.datatype, v_name)

            # Copy variable attributes
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})

            if "x" not in varin.dimensions:
                outVar[:] = varin[:]
            elif v_name == "nav_lon":
                outVar[:] = lons_t
            elif v_name == "nav_lat":
                outVar[:] = lats_t
            else:
                outVar[:] = interpolate_data_nn(varin[:], inds, lons_t.shape)



    # close the input file
    dsin.close()


if __name__ == '__main__':
    # main(in_file="/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_CC_drivenby_CRCM5_CanESM2_RCP85/EXP00/restarts_for_coupled_simulation/GLK_00157680_restart.nc",
    #      target_grid=known_domains["GLK_452x260_0.1deg"].to_gridconfig())


    main(in_file="/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_CC_drivenby_CRCM5_CanESM2_RCP85/EXP00/restarts_for_coupled_simulation/GLK_01734480_restart.nc",
         target_grid=known_domains["GLK_452x260_0.1deg"].to_gridconfig())

    # main(in_file="/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_CC_drivenby_CRCM5_CanESM2_RCP85/EXP00/restarts_for_coupled_simulation/GLK_00157680_restart_ice.nc",
    #      target_grid=known_domains["GLK_452x260_0.1deg"].to_gridconfig())
    #

    main(in_file="/RESCUE/skynet3_rech1/huziy/NEMO_OFFICIAL/dev_v3_4_STABLE_2012/NEMOGCM/CONFIG/GLK_LIM3_CC_drivenby_CRCM5_CanESM2_RCP85/EXP00/restarts_for_coupled_simulation/GLK_01734480_restart_ice.nc",
         target_grid=known_domains["GLK_452x260_0.1deg"].to_gridconfig())